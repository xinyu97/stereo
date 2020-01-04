# stereo

calibrate为单目相机标定，包括两部分，计算相机矩阵和反走样，将图片left.jpg与代码置于同一目录下运行得到结果，相机矩阵会直接输出；程序会新建output文件夹，反走样之后的图像将输出到该文件夹中。

Stereo calibrate为计算左右相机的变换矩阵，将图片与代码置于同一目录下，R,T输出到output.txt文件夹中。

Rectification进行的是极线平行校正，采用图片为left01.jpg和right01.jpg，输出结果为left_rectifying.jpg和right_rectifying.jpg
