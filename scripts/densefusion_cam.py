#%%
#! /usr/bin/env python3
from utils import *
import pyrealsense2 as rs

# clean terminal in the beginning
username = getpass.getuser()
osName = os.name
if osName == 'posix':
    os.system('clear')
else:
    os.system('cls')

# specify which gpu to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1" # "0,1,2,3"

num_objects = 1
num_points = 100

path = os.path.dirname(__file__)

pretrained_model = os.path.join(path + '/../txonigiri/', 'snapshot_model.npz')
pooling_func = cmr.functions.roi_align_2d
mask_rcnn = cmr.models.MaskRCNNResNet(
            n_layers=50,
            n_fg_class=1,
            pretrained_model= pretrained_model,
            pooling_func=pooling_func,
            anchor_scales=[4, 8, 16, 32],
            mean=(123.152, 115.903, 103.063),
            roi_size= 7, #7,14
            min_size=600,
            max_size=1000,
        )

chainer.cuda.get_device_from_id(0).use()
mask_rcnn.to_gpu()
print('maskrcnn model loaded %s' % pretrained_model)

pose = PoseNet(num_points, num_objects)
pose.cuda()
pose.load_state_dict(torch.load(path + '/../txonigiri/pose_model.pth'))
pose.eval()
print("pose model loaded...")

refiner = PoseRefineNet(num_points, num_objects)
refiner.cuda()
refiner.load_state_dict(torch.load(path + '/../txonigiri/pose_refine_model.pth'))
refiner.eval()
print("pose refine model loaded...")

filepath = (path + '/../txonigiri/txonigiri.ply')
mesh_model = o3d.io.read_triangle_mesh(filepath)
randomIndices = rand.sample(range(0, 9958), num_points)
print("object mesh model loaded...")

bs = 1
objId = 0
objlist =[1]
mm2m = 0.001
class_names = ['txonigiri']

# cam @ aist
cam_fx = 605.286
cam_cx = 320.075
cam_fy = 605.699
cam_cy = 247.877

dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
cam_mat = np.matrix([ [cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1] ])

edge = 60.
edge = edge * mm2m
edges  = np.array([
                [edge, -edge*.5,  edge],
                [edge, -edge*.5, -edge],
                [edge,  edge*.5, -edge],
                [edge,  edge*.5,  edge],
                [-edge,-edge*.5,  edge],
                [-edge,-edge*.5, -edge],
                [-edge, edge*.5, -edge],
                [-edge, edge*.5,  edge]])

#%%

class DenseFusion:

    def __init__(self, mask_rcnn, pose, refiner, object_index_, rgb, depth):

        self.rgb = rgb
        self.rgb = np.transpose(self.rgb, (2, 0, 1))

        self.rgb_s = []
        self.rgb_s.append(self.rgb)

        self.depth = depth

        self.mask_rcnn = mask_rcnn
        self.estimator = pose
        self.refiner = refiner
        self.object_index = object_index_

        """ publisher / subscriber """
        self.model_pub = rospy.Publisher("/rgbdCloud", PointCloud2, queue_size=30)
        self.pose_pub = rospy.Publisher('/onigiriPose', PoseArray, queue_size = 30)
        self.pose_sub = rospy.Subscriber('/onigiriPose', PoseArray, self.poseCallback, queue_size = 30)

        self.viz = np.zeros((480, 640, 3), np.uint8)
        self.objs_pose = None
        self.cloudPts = None
        self.poseArray = PoseArray()

        self.modelPts = np.asarray(mesh_model.vertices) * 0.01 #change units
        self.modelPts = self.modelPts[randomIndices, :]

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

    def poseCallback(self, poseArray):
        self.poseArray = poseArray

    def batch_predict(self):

        for batch in cmr.utils.batch(self.rgb_s, n=2):
            bboxes, masks, labels, scores = self.mask_rcnn.predict(batch)
            for bbox, mask, label, score in zip(bboxes, masks, labels, scores):
                yield bbox, mask, label, score

    def draw_seg(self, result):

        for img_chw, (bbox, mask, label, score) in zip(self.rgb_s, result):
            rgb = img_chw.transpose(1, 2, 0)
            del img_chw

            k = score >= 0.9
            bbox, mask, label, score = bbox[k], mask[k], label[k], score[k]
            i = np.argsort(score)
            bbox, mask, label, score = bbox[i], mask[i], label[i], score[i]

            captions = [
                '{}: {:.1%}'.format(class_names[l], s)
                for l, s in zip(label, score)
            ]
            # for caption in captions:
                # print(caption)

            viz = cmr.utils.draw_instance_bboxes(img=rgb, bboxes=bbox, labels=label + 1, n_class=len(class_names) + 1, captions=captions, masks=mask)

            # plt.imsave('seg_result/out/frame.png', viz)
            # plt.imshow(viz.copy()), plt.show()

        return (mask, bbox, viz)

    def pose_refiner(self, iteration, my_t, my_r, points, emb, idx):

        for ite in range(0, iteration):
            T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
            my_mat = quaternion_matrix(my_r)
            R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
            my_mat[0:3, 3] = my_t

            new_points = torch.bmm((points - T), R).contiguous()
            pred_r, pred_t = self.refiner(new_points, emb, idx)
            pred_r = pred_r.view(1, 1, -1)
            pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
            my_r_2 = pred_r.view(-1).cpu().data.numpy()
            my_t_2 = pred_t.view(-1).cpu().data.numpy()
            my_mat_2 = quaternion_matrix(my_r_2)
            my_mat_2[0:3, 3] = my_t_2
            # refine pose means two matrix multiplication
            my_mat_final = np.dot(my_mat, my_mat_2)
            my_r_final = copy.deepcopy(my_mat_final)
            my_r_final[0:3, 3] = 0
            my_r_final = quaternion_from_matrix(my_r_final, True)
            my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

            my_pred = np.append(my_r_final, my_t_final)
            my_r = my_r_final
            my_t = my_t_final

        return my_t, my_r

    def pose_estimator(self):

        t1 = time.time()
        obj_pose = []
        print(f"{Fore.GREEN}estimating pose..{Style.RESET_ALL}")

        """ mask rcnn """
        mask, bbox, viz = self.draw_seg(self.batch_predict())
        # cv2.imshow("mask", cv2.cvtColor(viz, cv2.COLOR_BGR2RGB)), cv2.moveWindow('mask', 0, 500)

        t2 = time.time()
        print(f'{Fore.YELLOW}mask-rcnn inference time is:{Style.RESET_ALL}', t2 - t1)

        pred = mask
        pred = pred *255
        pred = np.transpose(pred, (1, 2, 0)) # (CxHxW)->(HxWxC)

        # convert img into tensor
        rgb_original = self.rgb
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        all_masks = []
        # self.depth = depth.reshape(480, 640)
        mask_depth = ma.getmaskarray(ma.masked_not_equal(self.depth, 0))
        mask_label = ma.getmaskarray(ma.masked_equal(pred, np.array(255)))

        # iterate through detected masks
        for b in range(len(bbox)):

            mask = mask_depth * mask_label[:,:,b]
            rmin = int(bbox[b,0])
            rmax = int(bbox[b,1])
            cmin = int(bbox[b,2])
            cmax = int(bbox[b,3])

            img = np.transpose(rgb_original, (0, 1, 2)) #CxHxW
            choose = mask[rmin : rmax, cmin : cmax].flatten().nonzero()[0]
            if len(choose) == 0:
                cc = torch.LongTensor([0])
                return(cc, cc, cc, cc, cc, cc)

            if len(choose) > num_points:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:num_points] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')

            depth_masked = self.depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)

            choose = np.array([choose])
            choose = torch.LongTensor(choose.astype(np.int32))

            cam_scale = mm2m
            pt2 = depth_masked/cam_scale
            pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
            cloud = np.concatenate((pt0, pt1, pt2), axis=1)
            cloud = cloud /1000
            points = torch.from_numpy(cloud.astype(np.float32))

            img_masked = img[:, rmin : rmax, cmin : cmax ]
            img_ = norm(torch.from_numpy(img_masked.astype(np.float32)))
            idx = torch.LongTensor([self.object_index])
            img_ = Variable(img_).cuda().unsqueeze(0)
            points = Variable(points).cuda().unsqueeze(0)
            choose = Variable(choose).cuda().unsqueeze(0)
            idx = Variable(idx).cuda().unsqueeze(0)

            pred_r, pred_t, pred_c, emb = self.estimator(img_, points, choose, idx)
            pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
            pred_c = pred_c.view(bs, num_points)
            how_max, which_max = torch.max(pred_c, 1) #1
            pred_t = pred_t.view(bs * num_points, 1, 3)
            # print("max confidence", how_max)

            my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
            my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
            my_pred = np.append(my_r, my_t)

            """ DF refiner NOTE: results are better without refiner """
            # my_t, my_r = self.pose_refiner(1, my_t.T, my_r.T, points, emb, idx)

            """ get mean depth within a box as depth offset """
            depth = self.depth[rmin : rmax, cmin : cmax].astype(float)
            depth = depth * mm2m
            dep,_,_,_ = cv2.mean(depth)

            """ position mm2m """
            my_t = np.array(my_t*mm2m)
            my_t[2] = dep #NOTE: use this to get depth of obj centroid
            # print("Pos xyz:{0}".format(my_t))

            """ rotation """
            mat_r = quaternion_matrix(my_r)[:3, :3]
            # print('estimated rotation is\n:{0}'.format(mat_r))

            """ project point cloud """
            imgpts_cloud,_ = cv2.projectPoints(np.dot(points.cpu().numpy(), mat_r), mat_r, my_t, cam_mat, dist)
            viz = draw_pointCloud(viz, imgpts_cloud, [255, 0, 0])
            self.cloudPts = imgpts_cloud.reshape(num_points, 2)

            """ draw cmr 2D box """
            cv2.rectangle(viz, (cmax, cmin), (rmax, rmin), (255,0,0))

            """ introduce offset in Rot """
            Rx = rotation_matrix(2*m.pi/3, [1, 0, 0], my_t)
            Ry = rotation_matrix(10*m.pi/180, [0, 1, 0], my_t)
            Rz = rotation_matrix(5*m.pi/180, [0, 0, 1], my_t)
            offR = concatenate_matrices(Rx, Ry, Rz)[:3,:3]
            mat_r = np.dot(mat_r.T, offR[:3, :3])

            """ transform 3D box and axis with estimated pose and Draw """
            # target_df = np.dot(edges, mat_r) + my_t
            # new_image = cv2pil(viz)
            # g_draw = ImageDraw.Draw(new_image)
            # location, quaternion, projected_points = draw_cube(target_df, viz, g_draw, (255, 255, 255), cam_mat)
            # viz = pil2cv(new_image)

            """ convert pose to ros-msg """
            I = np.identity(4)
            I[0:3, 0:3] = mat_r
            I[0:3, -1] = my_t# + np.asarray(location) *0.01 #cm2m
            rot = quaternion_from_matrix(I, True) #wxyz
            my_t = my_t.reshape(1,3)
            pose = {
                    'tx':my_t[0][0],
                    'ty':my_t[0][1],
                    'tz':my_t[0][2],
                    'qw':rot[0],
                    'qx':rot[1],
                    'qy':rot[2],
                    'qz':rot[3]}

            obj_pose.append(pose)
            self.viz = viz
        else:
            if len(bbox) < 1:
                print(f"{Fore.RED}unable to detect pose..{Style.RESET_ALL}")

        self.objs_pose = obj_pose

        t3 = time.time()
        print(f'{Fore.YELLOW}DenseFusion inference time is:{Style.RESET_ALL}', t3 - t2)


def main():
    try:
        rospy.init_node('onigiriPose', anonymous=False)
        rospy.loginfo('streaming now...')
        # rospy.spin()
        # rate = rospy.Rate(10)
        # while not rospy.is_shutdown():
        #     rate.sleep()
        # except KeyboardInterrupt:
        #     print ('Shutting down densefusion ROS node')
        # cv2.destroyAllWindows()

        while True:

            t1 = time.time()

            """ Wait for frame (Color & Depth) """
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = rs.align(rs.stream.color).process(frames).get_depth_frame()

            if  not depth_frame or  not color_frame:
                raise ValueError('No image found, camera not streaming?')

            """ color image """
            rgb = np.asanyarray(color_frame.get_data())
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            """ Depth image """
            depth = np.asanyarray(depth_frame.get_data())

            """ merge depth with rgb NOTE:testing some feature -- not ready """
            gray_color = 153
            clipping_distance = 1 / mm2m # in mm
            depth_image_3d = np.dstack((depth, depth, depth))
            bg_removed = np.where((depth_image_3d >clipping_distance) | (depth_image_3d <= 0 ), gray_color, rgb)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha= 0.03 ), cv2.COLORMAP_JET)
            images = np.hstack((bg_removed, depth_colormap))
            # cv2.imshow("merged rgbd", cv2.cvtColor(images, cv2.COLOR_BGR2RGB))

            """ run DenseFusion """
            df = DenseFusion(mask_rcnn, pose, refiner, objId, rgb, depth)
            df.pose_estimator()

            """ point cloud """
            img = o3d.geometry.Image(np.asanyarray(color_frame.get_data()))
            dep = o3d.geometry.Image(np.asanyarray(depth_frame.get_data()))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img, dep)

            pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
            pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

            # http://www.open3d.org/docs/release/tutorial/Basic/working_with_numpy.html
            pcd = np.asarray(pcd.points)

            #NOTE: this will pause theloop
            # o3d.visualization.draw_geometries([pcd])

            """ publish to ros """
            Publisher(df.model_pub, df.pose_pub, cam_mat, dist,
                    df.viz, df.objs_pose, df.modelPts, pcd, "map")

            t2 = time.time()
            print('inference time is :{0}'.format(t2 - t1))

            key = cv2.waitKey(1) & 0xFF
            if  key == 27:
                print("stopping streaming...")
                break

            if t2-t0 > autostop:
                print("auto stop streaming after {} seconds".format(int(autostop)))
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':

    # Stream (Color/Depth) settings
    config = rs.config()
    config.enable_stream(rs.stream.color, 640 , 480 , rs.format.bgr8, 60)
    config.enable_stream(rs.stream.depth, 640 , 480 , rs.format.z16, 60)

    # Start streaming
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    t0 = time.time()

    autostop = 1000

    main()
# %%
