
# def init_feature(name):
#     chunks = name.split('-')
#     if chunks[0] == 'sift':
#         detector = cv2.SIFT_create()
#         norm = cv2.NORM_L2
#     elif chunks[0] == 'surf':
#         detector = cv2.SIFT_create(800)
#         norm = cv2.NORM_L2
#     elif chunks[0] == 'orb':
#         detector = cv2.ORB_create(400)
#         norm = cv2.NORM_HAMMING
#     elif chunks[0] == 'akaze':
#         detector = cv2.AKAZE_create()
#         norm = cv2.NORM_HAMMING
#     elif chunks[0] == 'brisk':
#         detector = cv2.BRISK_create()
#         norm = cv2.NORM_HAMMING
#     else:
#         return None, None
#     if 'flann' in chunks:
#         if norm == cv2.NORM_L2:
#             flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#         else:
#             flann_params= dict(algorithm = FLANN_INDEX_LSH,
#                                table_number = 6, # 12
#                                key_size = 12,     # 20
#                                multi_probe_level = 1) #2
#         matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
#     else:
#         matcher = cv2.BFMatcher(norm)
#     return detector, matcher


# def bring_horizontal_orientation(horizontal_dir_path, non_horizontal_dir_path):
#     h_im_path = sorted(os.listdir(horizontal_dir_path))[0]
#     h_im = cv2.imread(os.path.join(horizontal_dir_path,h_im_path))
#     h_im_g = cv2.cvtColor(h_im, cv2.COLOR_BGR2GRAY)
#
#     k=34
#     for nh_i_im_path in os.listdir(non_horizontal_dir_path)[k:k+4]:
#         full_nh_i_im_path = os.path.join(non_horizontal_dir_path, nh_i_im_path)
#         nh_i_im = cv2.imread(full_nh_i_im_path)
#         nh_i_im_g = cv2.cvtColor(nh_i_im, cv2.COLOR_BGR2GRAY)
#
#         # Detect keypoints (features) cand calculate the descriptors
#         _,axes = plt.subplots(4, 5, figsize=(700, 50))
#         for i,algo_n in enumerate(['orb','sift','surf', 'akaze','brisk']):
#             algo, matcher = init_feature(algo_n)
#             h_keypoints, h_descriptors = algo.detectAndCompute(h_im_g, None)
#             nh_i_keypoints, nh_i_descriptors = algo.detectAndCompute(nh_i_im_g, None)
#
#             # Match the keypoints
#             matches = matcher.match(h_descriptors, nh_i_descriptors)
#             matches = sorted(matches, key=lambda x: x.distance)[::]
#             pts_h_im = np.float32([h_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 2)
#             pts_nh_i_im = np.float32([nh_i_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 2)
#
#             # Draw the keypoint matches on the output image
#             output_img = cv2.drawMatches(h_im_g, h_keypoints,
#                                          nh_i_im_g, nh_i_keypoints, matches[:10], None)
#
#             output_img = cv2.resize(output_img,(4000,1200))
#             axes[0,i].imshow(output_img)
#             try:
#                 M0,m = cv2.findHomography(pts_nh_i_im,pts_h_im,cv2.RANSAC)
#                 M1 = cv2.getPerspectiveTransform(pts_nh_i_im[:4],pts_h_im[:4])
#                 M2 = cv2.getRotationMatrix2D(np.array(np.array(h_im_g.shape) // 2).astype('int8'),-43,1)
#                 h_i_im_g0 = cv2.warpPerspective(src=nh_i_im_g, M=M0, dsize=(h_im_g.shape[0], h_im_g.shape[1]))
#                 h_i_im_g1 = cv2.warpPerspective(src=nh_i_im_g, M=M1, dsize=(h_im_g.shape[0], h_im_g.shape[1]))
#                 h_i_im_g2 = cv2.warpAffine(src=nh_i_im_g, M=M2, dsize=(h_im_g.shape[0], h_im_g.shape[1]))
#                 fixed_img0 = cv2.drawMatches(h_im_g, [],
#                                             h_i_im_g0, [], [], None)
#                 fixed_img1 = cv2.drawMatches(h_im_g, [],
#                                             h_i_im_g1, [], [], None)
#                 fixed_img2 = cv2.drawMatches(h_im_g, [],
#                                             h_i_im_g2, [], [], None)
#                 axes[1, i].imshow(fixed_img0)
#                 axes[2, i].imshow(fixed_img1)
#                 axes[3, i].imshow(fixed_img2)
#             except:
#                 print(i,' bug')
#         plt.show()
#         # cv2.getPerspectiveTransform
