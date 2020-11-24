from __init__ import *

global t1
t1 = time.time()

def skew(R):
    return (0.5 * (R - R.T))

def cay(R):
    I = np.identity(4)
    return (np.linalg.inv((I - R)) * (I - R))

def pil2cv(image):
    new_image = np.array(image, dtype=np.uint8)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    return new_image

def cv2pil(image):
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    new_image = pImage.fromarray(new_image)
    return new_image

def makeRortho(R):
    ''' https://math.stackexchange.com/questions/3292034/normalizing-a-rotation-matrix '''
    return (cay(skew(cay(R))))

def draw_axis(img, R, t, K):
    # How+to+draw+3D+Coordinate+Axes+with+OpenCV+for+face+pose+estimation%3f
    rotV, _ = cv2.Rodrigues(R)
    points = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 3)
    return img

def draw_pointCloud(img, imgpts, color):

    for point in imgpts:
        img = cv2.circle(img,(int(point[0][0]),int(point[0][1])), 1, color, -1)
    return img

def draw_pointCloudList(img, imgpts, color):

    for c in range(len(imgpts)):
        for point in imgpts[c]:
            img = cv2.circle(img,(int(point[0][0]),int(point[0][1])), 1, color, -1)
    return img

def draw_cube(modelPts, viz, rvec, tvec, cam_mat):

    min_point = np.min(modelPts, axis=0)
    max_point = np.max(modelPts, axis=0)
    min_max = [[a,b] for a,b in zip(min_point, max_point)]
    #[[x_min, x_max], [y_min, y_max], [z_min, z_max]]

    vertices = itertools.product(*min_max)
    vertices = np.asarray(list(vertices))
    cuboid = cv2.projectPoints(vertices, rvec, tvec, cam_mat, None)[0]
    cuboid = np.transpose(np.asarray(cuboid), (1,0,2))[0]
    cuboid = [tuple(map(int, point)) for point in cuboid]

    line_width = 2
    cv2.line(viz, cuboid[0], cuboid[1], (255,255,255), line_width)
    cv2.line(viz, cuboid[0], cuboid[2], (255,255,255), line_width)
    cv2.line(viz, cuboid[0], cuboid[4], (255,255,255), line_width)
    cv2.line(viz, cuboid[1], cuboid[3], (255,255,255), line_width)
    cv2.line(viz, cuboid[1], cuboid[5], (255,255,255), line_width)
    cv2.line(viz, cuboid[2], cuboid[3], (255,255,255), line_width)
    cv2.line(viz, cuboid[2], cuboid[6], (255,255,255), line_width)
    cv2.line(viz, cuboid[3], cuboid[7], (255,255,255), line_width)
    cv2.line(viz, cuboid[4], cuboid[5], (255,255,255), line_width)
    cv2.line(viz, cuboid[4], cuboid[6], (255,255,255), line_width)
    cv2.line(viz, cuboid[5], cuboid[7], (255,255,255), line_width)
    cv2.line(viz, cuboid[6], cuboid[7], (255,255,255), line_width)

    cuboidCenter = (min_point + max_point)/2
    cuboidCenter = cv2.projectPoints(cuboidCenter, rvec, tvec, cam_mat, None)[0]
    cuboidCenter = np.transpose(np.asarray(cuboidCenter), (1,0,2))[0]
    for center in cuboidCenter:
        cv2.circle(viz, tuple(map(int, center)), 5, (0, 0, 0), -1)

    ''' PnP for refinement, adapted from DOPE '''
    pnpSolver = CuboidPNPSolver('txonigiri',
                                camera_intrinsic_matrix = cam_mat,
                                cuboid3d=Cuboid3d(size3d = [2., 2., 2.]))
    tvec, rvec, projPoints = pnpSolver.solve_pnp(cuboid)
    # rvec, tvec = cv2.solvePnPRefineVVS(modelPts, cloudPts, cam_mat, dist, rot_, my_t)
    draw_axis(viz, quaternion_matrix(rvec)[0:3, 0:3], np.array(tvec), cam_mat)

    return tvec, rvec, projPoints, center

def Publisher(model_pub, pose_pub, cam_mat, viz, objs_pose, modelPts, cloudPts, frame, method):

    global t1

    ''' publish rgbd cloud points '''
    headerPCD = std_msgs.msg.Header()
    headerPCD.stamp = rospy.Time.now()
    headerPCD.frame_id = frame
    scaled_cloud = PointCloud2()

    if frame == 'World': # NOTE: this has no effect when using realsense-ros pkg for rgb-d frames

        if method == 'open3d':
            pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(cloudPts, pinhole_camera_intrinsic)
            pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
            pcd.translate((0,0,1.5))

            pcdPts = np.asarray(pcd.points)
            sampleSize = int(len(pcdPts)/6)
            print('PCD downsampled from {} to {}'.format(len(pcdPts), sampleSize))
            downSamples = rand.sample(range(0, len(pcdPts)), sampleSize)
            pcdPts = pcdPts[downSamples, :]

            scaled_cloud = pcl2.create_cloud_xyz32(headerPCD, pcdPts)
            model_pub.publish(scaled_cloud)

            # to clear memory buffer
            t2 = time.time()
            if (int(t2-t1) % 10 == 0):
                pcdPts = np.zeros(shape=pcdPts.shape)
                scaled_cloud = pcl2.create_cloud_xyz32(headerPCD, pcdPts)
                model_pub.publish(scaled_cloud)
                del pcd

        if method == 'realsense':
            sampleSize = 50000
            downSamples = rand.sample(range(0, len(cloudPts)), sampleSize)
            cloudPts = cloudPts[downSamples, :]
            cloudPts = np.dot(cloudPts, [[1,0,0],[0,-1,0],[0,0,-1]]) #+ [(0,0,1.5)]
            scaled_cloud = pcl2.create_cloud_xyz32(headerPCD, cloudPts)
            model_pub.publish(scaled_cloud)

            # to clear memory buffer
            t2 = time.time()
            if (int(t2-t1) % 10 == 0):
                cloudPts = np.zeros(shape=cloudPts.shape)
                scaled_cloud = pcl2.create_cloud_xyz32(headerPCD, cloudPts)
                model_pub.publish(scaled_cloud)
                del cloudPts

    ''' publish pose to ros-msg '''
    if objs_pose is not None:

        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = frame
        poses = objs_pose
        print('total onigiri(s) found', len(poses))

        for p in range(len(poses)):

            pose2msg = Pose()
            pose2msg.position.x = poses[p]['tx']
            pose2msg.position.y = poses[p]['ty']
            pose2msg.position.z = poses[p]['tz']
            pose2msg.orientation.w = poses[p]['qw']
            pose2msg.orientation.x = poses[p]['qx']
            pose2msg.orientation.y = poses[p]['qy']
            pose2msg.orientation.z = poses[p]['qz']
            pose_array.poses.append(pose2msg)
            print(f'{Fore.RED} poseArray{Style.RESET_ALL}', pose_array.poses[p])

        pose_pub.publish(pose_array)

    else:
        print(f'{Fore.RED}Publisher: onigiri pose not detected{Style.RESET_ALL}')

    """ visualize pose """
    cv2.imshow('pose', cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))
    key = cv2.waitKey(1) & 0xFF # stop script
    if  key == 27:
        rospy.loginfo(f'{Fore.RED}stopping streaming...{Style.RESET_ALL}')
        try:
            sys.exit(1)
        except SystemExit:
            os._exit(0)