#! /usr/bin/env python3
from helper import *

# clean terminal in the beginning
username = getpass.getuser()
osName = os.name
if osName == 'posix':
    os.system('clear')
else:
    os.system('cls')

# specify which gpu to use
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0' # '0,1,2,3'

num_objects = 1
num_points = 100

path = os.path.dirname(__file__)

yolact_path = os.path.join(path + '/../txonigiri/', 'yolact_base_31999_800000.pth')
trained_model = SavePath.from_str(yolact_path)
set_cfg(trained_model.model_name + '_config')
cudnn.benchmark = True
cudnn.fastest = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')
yolact = Yolact()
yolact.load_weights(yolact_path)
yolact.eval()
yolact = yolact.cuda()
yolact.detect.use_fast_nms = True
yolact.detect.use_cross_class_nms = False
cfg.mask_proto_debug = False
print('yolact model loaded...')

pose = PoseNet(num_points, num_objects)
pose.cuda()
pose.load_state_dict(torch.load(path + '/../txonigiri/pose_modelv2.pth'))
pose.eval()
print('pose model loaded...')

refiner = PoseRefineNet(num_points, num_objects)
refiner.cuda()
refiner.load_state_dict(torch.load(path + '/../txonigiri/pose_refine_modelv2.pth'))
refiner.eval()
print('pose refine model loaded...')

filepath = (path + '/../txonigiri/txonigiri.ply')
mesh_model = o3d.io.read_triangle_mesh(filepath)
randomIndices = rand.sample(range(0, 266176), num_points)
print('object mesh model loaded...')

bs = 1
objId = 0
mm2m = 0.001
class_names = ['txonigiri']
color_cache = defaultdict(lambda: {})

# cam @ aist
cam_fx = 605.286
cam_cx = 320.075
cam_fy = 605.699
cam_cy = 247.877
cam_mat = np.matrix([ [cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1] ])

class DenseFusion:

    def __init__(self, yolact:Yolact, pose, refiner, object_index_):

        ''' publisher / subscriber '''
        rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/camera/depth/image_rect_raw', Image)
        self.model_pub = rospy.Publisher('/onigiriCloud', PointCloud2, queue_size = 30)
        self.pose_pub = rospy.Publisher('/onigiriPose', PoseArray, queue_size = 30)

        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub],
                                                            queue_size=1, slop=.1)
        # self.ts = message_filters.TimeSynchronizer([rgb_sub, depth_sub], 10)
        self.ts.registerCallback(self.callback)

        self.viz = np.zeros((480, 640, 3), np.uint8)
        self.objs_pose = None
        self.cloudPts = None

        self.yolact = yolact
        self.estimator = pose
        self.refiner = refiner
        self.object_index = object_index_

        self.modelPts = np.asarray(mesh_model.vertices) #* 0.01 #change units
        self.modelPts = self.modelPts[randomIndices, :]

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        # yolact specific
        self.fps = 0
        self.top_k = 100
        self.score_threshold = .7

        self.crop_masks = True
        self.display_masks = True
        self.display_scores = True
        self.display_bboxes = True
        self.display_text = True
        self.display_fps = False

        self.readyToSegmentAgain = True


    def callback(self, rgb, depth):

        depth = np.frombuffer(depth.data, dtype=np.uint16).reshape(depth.height, depth.width, -1)
        rgb = np.frombuffer(rgb.data, dtype=np.uint8).reshape(rgb.height, rgb.width, -1)

        ## visualize depth
        # convertDepth = depth.copy()
        # convertDepth = cv2.normalize(convertDepth, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # cv2.imshow("depth ros2numpy", convertDepth), cv2.waitKey(1)

        # initialize before referencing
        viz = rgb
        t1 = time.time()
        t2 = t1
        """ yolact """
        try:
            if self.readyToSegmentAgain:
                self.readyToSegmentAgain = False


                self.mask, self.bbox, viz = self.yolactMask(rgb)
                t2 = time.time()
                print(f'{Fore.YELLOW}---------------------yolact inference time is:{Style.RESET_ALL}', t2 - t1)
            else:
                # print('flag', self.readyToSegmentAgain)
                ''' estimate pose '''
                self.pose_estimator(rgb, depth, viz)
                t3 = time.time()
                print(f'{Fore.YELLOW}DenseFusion inference time is:{Style.RESET_ALL}', t3 - t2)

                ''' publish to ros '''
                Publisher(self.model_pub, self.pose_pub, cam_mat,
                        self.viz, self.objs_pose, self.modelPts, self.cloudPts,
                        'camera_depth_optical_frame', method=None)
        except rospy.ROSException:
            print(f'{Fore.RED}ROS Interrupted')

    def yolactMask(self, rgb):

        with torch.no_grad():
            frame = torch.from_numpy(rgb).cuda().float()
            batch = FastBaseTransform()(frame.unsqueeze(0))
            preds = self.yolact(batch)

            h, w, _ = frame.shape
            classes, scores, boxes, masks = self.postprocess_results(preds, w, h)
            print(len(boxes), f'{Fore.RED}onigiri(s) found{Style.RESET_ALL}')

            image = self.prep_display(classes, scores, boxes, masks, frame, fps_str=str(self.fps))
            cv2.imshow('yolact segmentation', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print('stopping, keyboard interrupt')
                # sys.exit()
                try:
                    sys.exit(1)
                except SystemExit:
                    os._exit(0)
        return masks, boxes, image

    def postprocess_results(self, dets_out, w, h):
            with timer.env('Postprocess'):
                save = cfg.rescore_bbox
                cfg.rescore_bbox = True
                t = postprocess(dets_out, w, h, visualize_lincomb = False,
                                                crop_masks        = self.crop_masks,
                                                score_threshold   = self.score_threshold)
                cfg.rescore_bbox = save

            with timer.env('Copy'):
                idx = t[1].argsort(0, descending=True)[:self.top_k]

                if cfg.eval_mask_branch:
                    # Masks are drawn on the GPU, so don't copy
                    masks = t[3][idx]
                classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

            return classes, scores, boxes, masks

    def prep_display(self, classes, scores, boxes, masks, img, class_color=False, mask_alpha=0.8, fps_str=''):

        img_gpu = img / 255.0

        num_dets_to_consider = min(self.top_k, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < self.score_threshold:
                num_dets_to_consider = j
                break

        # Quick and dirty lambda for selecting the color for a particular index
        # Also keeps track of a per-gpu color cache for maximum speed
        def get_color(j, on_gpu=None):
            global color_cache
            color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

            if on_gpu is not None and color_idx in color_cache[on_gpu]:
                return color_cache[on_gpu][color_idx]
            else:
                color = (100, 100, 255) #COLORS[color_idx] #BGR
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
                if on_gpu is not None:
                    color = torch.Tensor(color).to(on_gpu).float() / 255.
                    color_cache[on_gpu][color_idx] = color
                return color

        # First, draw the masks on the GPU where we can do it really fast
        # Beware: very fast but possibly unintelligible mask-drawing code ahead
        # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
        if self.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
            # After this, mask is of size [num_dets, h, w, 1]
            masks = masks[:num_dets_to_consider, :, :, None]

            # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
            colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
            masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

            # This is 1 everywhere except for 1-mask_alpha where the mask is
            inv_alph_masks = masks * (-mask_alpha) + 1

            # I did the math for this on pen and paper. This whole block should be equivalent to:
            #    for j in range(num_dets_to_consider):
            #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
            masks_color_summand = masks_color[0]
            if num_dets_to_consider > 1:
                inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
                masks_color_cumul = masks_color[1:] * inv_alph_cumul
                masks_color_summand += masks_color_cumul.sum(dim=0)

            img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

        if self.display_fps:
                # Draw the box for the fps on the GPU
            font_face = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            font_thickness = 1

            text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

            img_gpu[0:text_h+8, 0:text_w+8] *= 0.6 # 1 - Box alpha


        # Then draw the stuff that needs to be done on the cpu
        # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
        img_numpy = (img_gpu * 255).byte().cpu().numpy()

        if self.display_fps:
            # Draw the text on the CPU
            text_pt = (4, text_h + 2)
            text_color = [255, 255, 255]

            cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

        if num_dets_to_consider == 0:
            return img_numpy

        if self.display_text or self.display_bboxes:
            for j in reversed(range(num_dets_to_consider)):
                x1, y1, x2, y2 = boxes[j, :]
                color = get_color(j)
                score = scores[j]
                # print(f'{Fore.GREEN}at confidence{Style.RESET_ALL}', str(round(score*100)) + '%')

                if self.display_bboxes:
                    cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

                if self.display_text:
                    _class = cfg.dataset.class_names[classes[j]]
                    text_str = '%s: %.2f' % (_class, score) if self.display_scores else _class

                    font_face = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.6
                    font_thickness = 1

                    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                    text_pt = (x1, y1 - 3)
                    text_color = [255, 255, 255]

                    cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                    cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

        return img_numpy

    def pose_refiner(self, iteration, my_t, my_r, points, emb, idx):

        for ite in range(0, iteration):
            T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
            my_mat = quaternion_matrix(my_r)
            R = Variable(torch.from_numpy(my_mat[:3, :3].T.astype(np.float32))).cuda().view(1, 3, 3)
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

    def pose_estimator(self, rgb, depth, viz):

        obj_pose = []

        # print('before', len(self.bbox))
        if len(self.bbox) > 0:
            # print(f'{Fore.GREEN}estimating pose..{Style.RESET_ALL}')

            # print('total self.bbox', len(self.bbox))
            pred = self.mask.cpu().numpy()
            pred = pred #*255
            pred = np.transpose(pred, (1, 2, 0)) # (CxHxW)->(HxWxC)

            # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            # cv2.imshow('rgb_original', rgb), cv2.waitKey(1)

            rgb_original = np.transpose(rgb, (2, 0, 1))
            norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            self.depth = depth.reshape(480, 640)
            mask_depth = ma.getmaskarray(ma.masked_not_equal(self.depth, 0))
            # mask_label = ma.getmaskarray(ma.masked_equal(pred, np.array(255)))
            mask_label = ma.getmaskarray(ma.masked_equal(pred, np.array([0])))

            # print('mask_label', mask_depth.shape)
            # cv2.imshow('mask', np.uint8(mask_depth)), cv2.waitKey(1)

            # iterate through detected masks
            print('total objects found', len(self.bbox))
            for b in range(len(self.bbox)):

                print('objects processed', b)
                mask = mask_depth * mask_label[:,:,b]

                # convertDepth = cv2.normalize(np.uint8(mask), None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                # cv2.imshow('mask', convertDepth), cv2.waitKey(1)

                rmin = int(self.bbox[b,0])
                rmax = int(self.bbox[b,1])
                cmin = int(self.bbox[b,2])
                cmax = int(self.bbox[b,3])
                # print(rmin, rmax, cmin, cmax)

                # cv2.imshow('rgb_original', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)), cv2.waitKey(1)

                choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

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

                # convertDepth = cv2.normalize(np.uint8(depth_masked), None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                # cv2.imshow('depth masked', convertDepth), cv2.waitKey(1)

                choose = np.array([choose])
                choose = torch.LongTensor(choose.astype(np.int32))

                pt2 = depth_masked
                pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
                pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
                cloud = np.concatenate((pt0, pt1, pt2), axis=1)
                points = torch.from_numpy(cloud.astype(np.float32))

                # convert img into tensor
                img = np.transpose(rgb_original, (0, 1, 2)) #CxHxW
                img_masked = img[:, rmin:rmax, cmin:cmax ]
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
                # print('how_max', 'which_max', how_max, which_max)

                my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
                my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
                my_pred = np.append(my_r, my_t)

                ''' DF refiner NOTE: results are better without refiner '''
                my_t, my_r = self.pose_refiner(1, my_t, my_r, points, emb, idx)

                ''' offset (mm) to align with obj-center '''
                # get mean depth within a bbox as new object depth
                meandepth = self.depth[rmin:rmax, cmin:cmax].astype(float)

                # remove NAN and Zeros before taking depth mean
                nonZero = meandepth[np.nonzero(meandepth)]
                nonNaNDepth = np.nanmean(nonZero)
                my_t[2] = nonNaNDepth
                my_t[0] = my_t[0] + 20
                my_t[1] = my_t[1] - 20

                ''' position mm2m '''
                my_t = np.array(my_t*mm2m)
                # print('Pos xyz:{0}'.format(my_t))

                ''' rotation '''
                mat_r = quaternion_matrix(my_r)[:3, :3]
                # mat_r = mat_r.transpose()
                # print('estimated rotation is\n:{0}'.format(mat_r))

                ''' project depth point cloud '''
                imgpts_cloud, jac = cv2.projectPoints(np.dot(points.cpu().numpy(), mat_r), mat_r, my_t, cam_mat, None)
                viz = draw_pointCloud(viz, imgpts_cloud, [255, 0, 0]) # cloudPts
                self.cloudPts = imgpts_cloud.reshape(num_points, 2)

                ''' draw cmr 2D box '''
                cv2.rectangle(viz, (cmax, cmin), (rmax, rmin), (255,0,255), 2)

                ''' project the 3D bounding-box to 2D image plane '''
                tvec, rvec, projPoints, center = draw_cube(self.modelPts, viz, mat_r, my_t, cam_mat)
                # print('center', center)

                ''' add PnP to DF's predicted pose'''
                I = np.identity(4)
                I[0:3, 0:3] = np.dot(mat_r.T, quaternion_matrix(rvec)[0:3, 0:3]) # mat_r
                I[0:3, -1] =  my_t + np.asarray(tvec) *mm2m # my_t
                rot = quaternion_from_matrix(I, True) #wxyz

                ''' convert pose to ros-msg '''
                my_t = I[0:3, -1]
                my_t = my_t.reshape(1,3)
                pose = {
                        'tx':my_t[0][0],
                        'ty':my_t[0][1],
                        'tz':my_t[0][2],
                        'qw':rot[0],
                        'qx':rot[1],
                        'qy':rot[2],
                        'qz':rot[3]}

                '''publish only if predicted pose is within the respective bbox'''
                # if  (min(cmax,rmax) < center[0] and center[0] < max(cmax,rmax)) and \
                #     (min(cmin,rmin) < center[1] and center[1] < max(cmin,rmin)):
                obj_pose.append(pose)
                self.viz = viz

            print('total obj processed', len(obj_pose))
            self.objs_pose = obj_pose

        if  b == len(self.bbox):
            self.readyToSegmentAgain = True
            print(f'{Fore.RED}processing next segmentation sample{Style.RESET_ALL}')

        else:
            # self.readyToSegmentAgain = True
            print(f'{Fore.RED}no mask or onigiri detected? {Style.RESET_ALL}')


def main():

    rospy.init_node('onigiriPose', anonymous=False)
    rospy.loginfo('starting onigiriPose node...')

    ''' run DenseFusion '''
    DenseFusion(yolact, pose, refiner, objId)

    try:
        rospy.spin()
        rate = rospy.Rate(0)
        while not rospy.is_shutdown():
            rate.sleep()
    except KeyboardInterrupt:
        print ('Shutting down densefusion ROS node')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
