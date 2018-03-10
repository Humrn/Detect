#ifndef DETECT_H
#define DETECT_H

#include<iostream>
#include"opencv2/core.hpp"
#include"opencv2/highgui.hpp"
#include"opencv2/video.hpp"
#include"opencv2/imgproc.hpp"
#include "omp.h"
#include<time.h>


using namespace std;
using namespace cv;

////宏定义
//#define T_ANGLE_THRE 15   //角度阈值，两个LED灯的角度差的绝对值所允许的最大值
//#define T_SIZE_THRE  5    //尺寸阈值,值越大能识别的数量就越少
//#define T_LED_COLOR  2    //代表要识别的装甲的颜色，1代表蓝色，2代表红色,3代表都识别


//要识别的小车的LED灯颜色
enum DetectColor {
	Red=1,
	Blue=2,
	RedAndBlue=3
};


class Detect
{
public:
	Detect();
	Detect(String videoPath);
	~Detect();
	//函数声明
	void                showAdjustUGI();//显示参数调节界面
	void                closeAdjustUGI();//关闭参数调节界面
	void                optimizeBinary(Mat src, Mat& binaryImg, Mat& dst, int nSize);//优化二值图，减低模糊边缘造成的干扰
	void                brightAdjust(Mat src, Mat dst, double dContrast, double dBright);//亮度调节函数
	void                getDiffImage(Mat src1, Mat src2, Mat dst, int nThre,DetectColor t_led_color=Red);//二值化
	vector<RotatedRect> armorDetect(Mat src, vector<RotatedRect> vEllipse);//检测装甲位置
	void                drawBox(RotatedRect box, Mat img);//标记出装甲矩形
	Scalar              findColorThre(Mat src, Scalar & minColorThre, Scalar & maxColorThre);//计算一个模板的颜色最小阈值和最大阈值，并返回平均值
	RotatedRect         getAttackTarget();//获取攻击目标
public:
	int                 T_ANGLE_THRE = 15; //角度阈值，两个LED灯的角度差的绝对值所允许的最大值
	int                 T_SIZE_THRE = 4; //尺寸阈值,值越大能识别的数量就越少
	int                 lowDifference = 4;//漫水填充的负差最大值和正差最大值，默认都是3
	int                 upDifference = 3;
	int                 brightThre = 120;//亮度调节阈值
	int                 br = 25;//LED的变换的g值与r值之差阈值
	int                 dilateSize = 3;//膨胀的核大小
	int                 erodeSize = 1;//腐蚀的和大小
	int                 isShowFloorFill = 0;//是否显示漫水效果
	int                 floorFillThre = 5;//通过漫水填充的阈值，值越小，通过的机率越大，能识别出来的矩形越多
	String              videoPath="";//读入图像的路径，若为空，在图片来源为设备的摄像头
	String              ledPath = "";//灯的模板的路径，若为空，则不适用LED灯模板
	DetectColor         T_LED_COLOR = Red;//代表要识别的装甲的颜色,默认为红色
	Scalar              armorCenterColorThre=Scalar(70,70,70);//装甲矩阵的中心领域的颜色阈值，默认为rgb值都小于50
	Size                frameSize = Size(800, 600);//读取图片的大小
private:
	String              winName = "参数调节界面";
	double              resizeW = 0.4;//图像宽的放缩比例,默认为0.4
	double              resizeH = 0.5;//图像高的放缩比例，默认为0.5
	Scalar              minColorThre = Scalar(20, 20, 20);//默认LED灯颜色范围，使用LED模板的时候才有用
	Scalar              maxColorThre = Scalar(250, 250, 250);
};
#endif
