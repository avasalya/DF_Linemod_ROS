from __init__ import *

def skew(R):
    return (0.5 * (R - R.T))

def cay(R):
    I = np.identity(4)
    return (np.linalg.inv((I - R)) * (I - R))

def makeRortho(R):
    """ https://math.stackexchange.com/questions/3292034/normalizing-a-rotation-matrix """
    return (cay(skew(cay(R))))

def draw_axis(img, R, t, K):
    # How+to+draw+3D+Coordinate+Axes+with+OpenCV+for+face+pose+estimation%3f
    rotV, _ = cv2.Rodrigues(R)
    points = np.float32([[.1, 0, 0], [0, .1, 0], [0, 0, .1], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 3)
    return img

def draw_pointCloud(img, imgpts, color):
    for point in imgpts:
        img=cv2.circle(img,(int(point[0][0]),int(point[0][1])), 1, color, -1)
    return img

def pil2cv(image):
    new_image = np.array(image, dtype=np.uint8)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    return new_image

def cv2pil(image):
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    new_image = pImage.fromarray(new_image)
    return new_image

""" adapted from DOPE """
def DrawLine(g_draw, point1, point2, lineColor, lineWidth):
    '''Draws line on image'''
    if not point1 is None and point2 is not None:
        g_draw.line([point1, point2], fill=lineColor, width=lineWidth)

def DrawDot(g_draw, point, pointColor, pointRadius):
    '''Draws dot (filled circle) on image'''
    if point is not None:
        xy = [
            point[0] - pointRadius,
            point[1] - pointRadius,
            point[0] + pointRadius,
            point[1] + pointRadius
        ]
        g_draw.ellipse(xy,
                    fill=pointColor,
                    outline=pointColor
                    )

def draw_cube(tar, img, g_draw, color, cam_mat):

    cam_fx = cam_mat[0,0]
    cam_cx = cam_mat[0,2]
    cam_fy = cam_mat[1,1]
    cam_cy = cam_mat[1,2]

    # pinhole camera model/ project the cubeoid on the image plane using camera intrinsics
    # u = fx * x/z + cx
    # v = fy * y/z + cy
    # https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga50620f0e26e02caa2e9adc07b5fbf24e
    p0 = (int((tar[0][0]/ tar[0][2])*cam_fx + cam_cx),  int((tar[0][1]/ tar[0][2])*cam_fy + cam_cy))
    p1 = (int((tar[1][0]/ tar[1][2])*cam_fx + cam_cx),  int((tar[1][1]/ tar[1][2])*cam_fy + cam_cy))
    p2 = (int((tar[2][0]/ tar[2][2])*cam_fx + cam_cx),  int((tar[2][1]/ tar[2][2])*cam_fy + cam_cy))
    p3 = (int((tar[3][0]/ tar[3][2])*cam_fx + cam_cx),  int((tar[3][1]/ tar[3][2])*cam_fy + cam_cy))
    p4 = (int((tar[4][0]/ tar[4][2])*cam_fx + cam_cx),  int((tar[4][1]/ tar[4][2])*cam_fy + cam_cy))
    p5 = (int((tar[5][0]/ tar[5][2])*cam_fx + cam_cx),  int((tar[5][1]/ tar[5][2])*cam_fy + cam_cy))
    p6 = (int((tar[6][0]/ tar[6][2])*cam_fx + cam_cx),  int((tar[6][1]/ tar[6][2])*cam_fy + cam_cy))
    p7 = (int((tar[7][0]/ tar[7][2])*cam_fx + cam_cx),  int((tar[7][1]/ tar[7][2])*cam_fy + cam_cy))

    """ PnP for refinement """
    points2D = [list(p0), list(p1), list(p2), list(p3), list(p4), list(p5), list(p6), list(p7)]

    pnpSolver = CuboidPNPSolver('txonigiri',
                                camera_intrinsic_matrix = cam_mat,
                                cuboid3d=Cuboid3d(size3d = [4., 4., 4.]))
    location, quaternion, projected_points = pnpSolver.solve_pnp(points2D)
    # rvec, tvec = cv2.solvePnPRefineVVS(modelPts, cloudPts, cam_mat, dist, rot_, my_t)

    points = []
    points.append(tuple(projected_points[0]))
    points.append(tuple(projected_points[1]))
    points.append(tuple(projected_points[2]))
    points.append(tuple(projected_points[3]))
    points.append(tuple(projected_points[4]))
    points.append(tuple(projected_points[5]))
    points.append(tuple(projected_points[6]))
    points.append(tuple(projected_points[7]))

    '''
    Draws cube with a thick solid line across
    the front top edge and an X on the top face.
    '''
    lineWidthForDrawing = 2

    # draw front
    DrawLine(g_draw, points[0], points[1], color, lineWidthForDrawing)
    DrawLine(g_draw, points[1], points[2], color, lineWidthForDrawing)
    DrawLine(g_draw, points[3], points[2], color, lineWidthForDrawing)
    DrawLine(g_draw, points[3], points[0], color, lineWidthForDrawing)

    # draw back
    DrawLine(g_draw, points[4], points[5], color, lineWidthForDrawing)
    DrawLine(g_draw, points[6], points[5], color, lineWidthForDrawing)
    DrawLine(g_draw, points[6], points[7], color, lineWidthForDrawing)
    DrawLine(g_draw, points[4], points[7], color, lineWidthForDrawing)

    # draw sides
    DrawLine(g_draw, points[0], points[4], color, lineWidthForDrawing)
    DrawLine(g_draw, points[7], points[3], color, lineWidthForDrawing)
    DrawLine(g_draw, points[5], points[1], color, lineWidthForDrawing)
    DrawLine(g_draw, points[2], points[6], color, lineWidthForDrawing)

    # draw dots
    DrawDot(g_draw, points[0], pointColor=(0,101,255), pointRadius=4)
    DrawDot(g_draw, points[1], pointColor=(232,12,217), pointRadius=4)

    # draw x on the top
    DrawLine(g_draw, points[0], points[5], color, lineWidthForDrawing)
    DrawLine(g_draw, points[1], points[4], color, lineWidthForDrawing)


    r = 255 # int(np.random.choice(range(255)))
    g = 255 # int(np.random.choice(range(255)))
    b = 255 # int(np.random.choice(range(255)))

    # cv2.line(img, p0, p1, (0,0,b), 2)
    # cv2.line(img, p0, p3, (r,0,0), 2)
    # cv2.line(img, p0, p4, (0,g,0), 2)

    # cv2.line(img, p0, p1, color, 2)
    # cv2.line(img, p0, p3, color, 2)
    # cv2.line(img, p0, p4, color, 2)
    # cv2.line(img, p1, p2, color, 2)
    # cv2.line(img, p1, p5, color, 2)
    # cv2.line(img, p2, p3, color, 2)
    # cv2.line(img, p2, p6, color, 2)
    # cv2.line(img, p3, p7, color, 2)
    # cv2.line(img, p4, p5, color, 2)
    # cv2.line(img, p4, p7, color, 2)
    # cv2.line(img, p5, p6, color, 2)
    # cv2.line(img, p6, p7, color, 2)

    # print(p0, p1, p2, p3, p4, p5, p6, p7)
    # cv2.rectangle(img, p0, p7, (0,0,255))

    return location, quaternion, projected_points

def Publisher(model_pub, pose_pub, cam_mat, dist, viz, objs_pose, modelPts, rgbd, frame):

    """ publish model cloud points """
    headerPCD = std_msgs.msg.Header()
    headerPCD.stamp = rospy.Time.now()
    headerPCD.frame_id = frame
    scaled_cloud = PointCloud2()

    """ publish pose to ros-msg """
    pose2msg = Pose()
    pose_array = PoseArray()
    pose_array.header.stamp = rospy.Time.now()
    pose_array.header.frame_id = frame
    poses = objs_pose

    if poses is not None:

        print("total onigiri(s) found", len(poses))
        for p in range(len(poses)):
            print(str(p), poses[p])
            pose2msg.position.x = poses[p]['tx']
            pose2msg.position.y = poses[p]['ty']
            pose2msg.position.z = poses[p]['tz']
            pose2msg.orientation.w = poses[p]['qw']
            pose2msg.orientation.x = poses[p]['qx']
            pose2msg.orientation.y = poses[p]['qy']
            pose2msg.orientation.z = poses[p]['qz']
            pose_array.poses.append(pose2msg)

            pos = np.array([poses[p]['tx'], poses[p]['ty'], poses[p]['tz']])
            q2rot = quaternion_matrix([poses[p]['qw'], poses[p]['qx'], poses[p]['qy'], poses[p]['qz']])

            """ transform modelPoints w.r.t estimated pose """
            modelPts = np.dot(modelPts, q2rot[0:3, 0:3]) + pos

            """ send to ros """
            if frame == "map":
                pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)

                pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
                pcd.translate((0,0,2))

                pcdPts = np.asarray(pcd.points)
                print("PCD actual size", pcdPts.shape)

                sampleSize = 50000
                downSamples = rand.sample(range(0, len(pcdPts)), sampleSize)
                pcdPts = pcdPts[downSamples, :]
                print("PCD downsampled to", pcdPts.shape)
                scaled_cloud = pcl2.create_cloud_xyz32(headerPCD, pcdPts)

                #NOTE: this will pause the loop -- use only for debugging
                # o3d.visualization.draw_geometries([pcdPts])

            else:
                scaled_cloud = pcl2.create_cloud_xyz32(headerPCD, modelPts)

            model_pub.publish(scaled_cloud)

            """ publish/visualize pose """
            if viz is not None:
                imgpts_cloud,jac = cv2.projectPoints(modelPts, np.identity(3), np.array([0.,0.,0.]), cam_mat, dist)
                vizPnP = draw_pointCloud(viz, imgpts_cloud, [0,255,0]) # modelPts
                draw_axis(viz, q2rot[0:3, 0:3], pos, cam_mat)
                modelPts = np.zeros(shape=modelPts.shape)
                cv2.imshow("posePnP", cv2.cvtColor(vizPnP, cv2.COLOR_BGR2RGB))

            key = cv2.waitKey(1) & 0xFF
            if  key == 27:
                rospy.loginfo(f"{Fore.RED}stopping streaming...{Style.RESET_ALL}")
                try:
                    sys.exit(1)
                except SystemExit:
                    os._exit(0)

            print(f"{Fore.RED} poseArray{Style.RESET_ALL}", pose_array.poses[p])
        pose_pub.publish(pose_array)
    else:
        print(f"{Fore.RED}no onigiri detected{Style.RESET_ALL}")
