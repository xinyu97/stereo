# stereo

calibrate为单目相机标定，包括两部分，计算相机矩阵和反走样，将图片left.jpg与代码置于同一目录下运行得到结果，相机矩阵会直接输出；程序会新建output文件夹，反走样之后的图像将输出到该文件夹中。

Stereo calibrate为计算左右相机的变换矩阵，将图片与代码置于同一目录下，R,T输出到output.txt文件夹中。根据上一部分代码，分别求出left和right图像的相机矩阵，旋转向量和平移向量，并进行反走样之后得到将畸变的影响消除后的相机矩阵。然后运用函数cv2.Rodrigues 从旋转向量得到旋转矩阵,然后根据题干中博客的公式，推得每一组图片的相机的变换矩阵(R|t),最后将结果输出到output.txt中

Rectification进行的是极线平行校正，采用图片为left01.jpg和right01.jpg，输出结果为left_rectifying.jpg和right_rectifying.jpg。用cv2.stereoRectify 得到P1,R1,P2,R2,使用cv2.initUndistortRectifyMap计算结果，用cv2.remap将图形变换，一开始输入的相机矩阵为反走样之前的相机矩阵，得到的结果left图片十分扭曲，right 图片为全黑,之后将反走样之后的相机矩阵输入，仍然得不到目标的图像。我输出了每一步的结果，cv2.initUndistortRectifyMap的输出结果中，最后面的数为负的20000多，我怀疑是这步或之前出了问题，但我没有找到解决的办法。我考虑了是否应该选取其他函数，然在github 找了几个项目均采用的这几个函数，而且发现官方文档中也是如此使用。之后在问答区域找到了与我情况类似的提问，但他的解答我也没有看懂，是将矩阵P人为设定为一个值，有解答的提问看了2页，没有找到自己需要的。我尝试从理论上推导得到P1,P2后图像应该变成什么样，rectify变化由P1,P2唯一确定，也就是说在计算得到P1,P2之后，可以唯一确认原图像所有像素点在变换之后的位置，但矩阵$T=P_nP_o^{-1}$中原外参矩阵未知，需要用左右图像中对应点求解。

SGBM为采用OpenCV中的SGBM算法，计算经过极线平行校正后的左右视图的视差图。
