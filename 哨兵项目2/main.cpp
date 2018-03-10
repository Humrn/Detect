#include<iostream>
#include"opencv2/core.hpp"
#include"opencv2/highgui.hpp"
#include"opencv2/video.hpp"
#include"opencv2/imgproc.hpp"
#include "omp.h"
#include<time.h>


using namespace std;
using namespace cv;

//宏定义
#define T_ANGLE_THRE 15   //角度阈值
#define T_SIZE_THRE  5     //尺寸阈值
#define T_LED_COLOR  2    //代表要识别的装甲的颜色，1代表蓝色，2代表红色,3代表都识别

//函数声明
void                brightAdjust(Mat src, Mat dst, double dContrast, double dBright);//亮度调节函数
void                getDiffImage(Mat src1, Mat src2, Mat dst, int nThre);//二值化
vector<RotatedRect> armorDetect(Mat src, vector<RotatedRect> vEllipse);//检测装甲位置
void                drawBox(RotatedRect box, Mat img);//标记出装甲矩形
Scalar              findColorThre(Mat src, Scalar & minColorThre, Scalar & maxColorThre);//计算一个模板的颜色最小阈值和最大阈值，并返回平均值
//全局变量声明
double resizeW = 0.4;//图像宽的放缩比例,默认为0.4
double resizeH = 0.5;//图像高的放缩比例，默认为0.5
Size frameSize=Size(800,600);//读取图片的大小
Scalar minColorThre=Scalar(20,20,20);//默认LED灯颜色范围
Scalar maxColorThre=Scalar(250,255,255);
int lowDifference = 3;//漫水填充的负差最大值和正差最大值
int upDifference = 3;
int brightThre = -120;//亮度调节阈值
int gr = 60;

void main1()
{
	////模板处理
	//Mat LEDcenterImg = imread("yellow4.jpg");//读入LED灯中心区域模板图
	//findColorThre(LEDcenterImg, minColorThre, maxColorThre);//计算LED灯中心区域颜色阈值
	//cout << "minB：" << minColorThre.val[0] << "   minG:" << minColorThre.val[1] << "  minR:" << minColorThre.val[2] << endl;
	//cout << "maxB：" << maxColorThre.val[0] << "   maxG:" << maxColorThre.val[1] << "  maxR:" << maxColorThre.val[2] << endl;
	//
	
	//读入图像识别目标
	VideoCapture myCamera("步兵素材红车旋转-ev-0.MOV");
	myCamera.set(CV_CAP_PROP_FRAME_WIDTH, 300);
	myCamera.set(CV_CAP_PROP_FRAME_HEIGHT, 300);
    Mat resizeImg,srcImg;
	//Mat srcImg = imread("chezi.png");
	myCamera >> srcImg;//获取一张图片
    //将图片缩小后再处理，节省处理时间
	resize(srcImg, resizeImg,frameSize,0,0,INTER_NEAREST);
	//计算放缩比
	resizeW = frameSize.width / srcImg.cols;
	resizeH = frameSize.height / srcImg.rows;
    //imshow("原图", srcImg);
	Size imgSize;
	RotatedRect s;//定义旋转矩形
	vector<RotatedRect> vEllipse;//定于旋转矩阵的向量，用于存储发现的目标区域轮廓
	vector<RotatedRect> vRlt;//装甲旋转矩形向量
	vector<RotatedRect> vArmor;
	bool bFlag = false;
	vector<vector<Point>> contour;//用于存储检测到的轮廓
	imgSize = resizeImg.size();

	//定义一些缓冲数组，用于处理图像
	Mat rawImg = Mat(imgSize, resizeImg.type());//存储亮度调整后的图片
	
	Mat grayImg = Mat(imgSize, CV_8UC1);
	Mat rImg = Mat(imgSize, CV_8UC1);
	Mat gImg = Mat(imgSize, CV_8UC1);
	Mat bImg = Mat(imgSize, CV_8UC1);
	Mat binaryImg = Mat(imgSize, CV_8UC1);
	Mat rltImg = Mat(imgSize, CV_8UC1); 
	
	while (1)
	{
		if (myCamera.read(srcImg))//获取到了一张图片
		{
			//将图片缩小后再处理，节省处理时间
			resize(srcImg, resizeImg,frameSize, 0, 0, INTER_NEAREST);
			double beginTime = getTickCount();
			vector<Mat> bgr;
			brightAdjust(resizeImg, rawImg, 1, brightThre);//调整图片亮度，突出LED灯
			split(rawImg, bgr);//将三个通道的像素分离
			bImg = bgr[0];
			gImg = bgr[1];
			rImg = bgr[2];
			//如果像素R值与G值得之差大于25，则返回的二值图像的值为255，否则为0
			if (T_LED_COLOR == 1)//目标为蓝色
			{
				getDiffImage(bImg, rImg, binaryImg, gr);
			}
			else {//目标为红色
				getDiffImage(rImg, bImg, binaryImg, gr);
			}
			imshow("二值图", binaryImg);
			dilate(binaryImg, grayImg, Mat(), Point(-1, -1), 3);//图像膨胀
			erode(grayImg, rltImg, Mat(), Point(-1, -1), 1);//图像腐蚀，先膨胀再腐蚀属于闭运算
			imshow("闭运算图", rltImg);
			contour.clear();
			findContours(rltImg, contour, RETR_CCOMP, CHAIN_APPROX_SIMPLE);//在二值图像中寻找轮廓
			for (int i = 0; i < contour.size(); i++)
			{
				if (contour[i].size()>4)//判断当前轮廓是否大于10个像素点
				{
					bFlag = true;//如果大于10个，则检测到了目标区域
					//拟合目标区域成为椭圆，返回一个选择矩形
					s = fitEllipse(contour[i]);
					for (int nI = 0; nI < 1; nI++)//遍历以旋转矩形中心点为中心的1*1的像素块
					{
						for (int nJ = 0; nJ < 1; nJ++)
						{
							int x = s.center.x - 0 + nJ;
							int y = s.center.y - 0 + nI;
							if (x > 0 && x < resizeImg.cols&&y>0 && y < resizeImg.rows)
							{
								Vec3b v3b = resizeImg.at<Vec3b>(y, x);
								//如果中心区域不接近模板颜色，侧不是LED轮廓
								if (v3b[0] <minColorThre.val[0] - 30 || v3b[1] < minColorThre.val[1] - 30 || v3b[2] < minColorThre.val[2] - 30)
									bFlag = false;
								if (v3b[0] > maxColorThre.val[0] + 10 || v3b[1] > maxColorThre.val[1] + 10 || v3b[2]>maxColorThre.val[2] + 10)
									bFlag = false;
							}
						}
					}
					if (bFlag)
					{
						vEllipse.push_back(s);
					}

				}
			}
			//画出检测到的LED灯旋转矩形
			for (int i = 0; i < vEllipse.size();i++)
			{
				drawBox(vEllipse[i], rawImg);
			}
			//调用子程序，在输入的LED所在的旋转矩形的vector中找出装甲的位置，并包装成旋转矩形，存入vector并返回
			vRlt = armorDetect(resizeImg,vEllipse);
			for (unsigned int i = 0; i < vRlt.size(); i++)
			{
				drawBox(vRlt[i], resizeImg);
			}
			imshow("亮度调整效果图", rawImg);
			imshow("缩小后的效果图", resizeImg);
			////求出原图的装甲所在的矩形
			//for (unsigned int i = 0; i < vRlt.size(); i++)
			//{
			//	double rsX = 1.0 / resizeW;
			//	double rsY = 1.0 / resizeH;
			//	vRlt[i].center.x *= rsX;
			//	vRlt[i].center.y *= rsY;
			//	vRlt[i].size.height *= rsY;
			//	vRlt[i].size.width *= rsX;
			//	drawBox(vRlt[i], srcImg);
			//}
			//imshow("效果图", srcImg);

			//清空数组
			vEllipse.clear();
			vRlt.clear();
			vArmor.clear();
			cout << "处理时间：" << (getTickCount() - beginTime) / getTickFrequency() * 1000 << "ms" << endl;
			char pressedKey = waitKey(1);
			if (pressedKey == 27)
				break;
			if (pressedKey=='1')
			{
				continue;
			}
		}
		else {//没有可获取的图片了
			break;
		}
	}

	waitKey(0);
	system("pause");
}

//调整亮度,突出LED灯
void brightAdjust(Mat src, Mat dst, double dContrast, double dBright)
{
	int nVal;
	if (dst.empty())//如果目标图还没有分配内存，则分配内存
	{
		dst = Mat(src.size(), src.type());
	}
//	//开8个并行线程，执行下面for循环加快运行速度
//	omp_set_num_threads(8);
//#pragma omp parallel for

	for (int i = 0; i < src.rows;i++)
	{
		Vec3b* p1 = src.ptr<Vec3b>(i);//获取原图的行指针
		Vec3b* p2 = dst.ptr<Vec3b>(i);//获取目标图的行指针
//
		for (int j = 0; j < src.cols; j++)
		{

			for (int k = 0; k < 3;k++)
			{
//#pragma omp critical
				//对每个像素的每个通道的值都进行线性变换
				nVal = (int)(dContrast*p1[j][k] + dBright);
				if (nVal < 0)
					nVal = 0;
				if (nVal > 255)
					nVal = 255;
				p2[j][k] = nVal;
			}
		}
	}
}

//图像二值化，为轮廓检测做准备
void getDiffImage(Mat src1, Mat src2, Mat dst, int nThre)
{
	if (dst.empty())
	{
		throw"目标图像没有内存空间，请初始化！";
	}
	omp_set_num_threads(8);
#pragma omp parallel for
	for (int i = 0; i < src1.rows;i++)
	{
		//获取图像的行指针
		uchar * pchar1 = src1.ptr<uchar>(i);
		uchar * pchar2 = src2.ptr<uchar>(i);
		uchar * pchar3 = dst.ptr<uchar>(i);
		for (int j = 0; j < src1.cols;j++)
		{
			if (pchar1[j]-pchar2[j]>nThre)
			{
				pchar3[j] = 255;
			}
			else
			{
				pchar3[j] = 0;
			}
		}
	}
}

//检测装甲位置
vector<RotatedRect> armorDetect(Mat src,vector<RotatedRect> vEllipse)
{
	vector<RotatedRect> vRlt;
	Mat   ROI_img;//旋转矩形感兴趣局域
	RotatedRect armor;//定义装甲局域的旋转矩形
	int nL, nW;
	double dAngle;
	vRlt.clear();//清空装甲局域的旋转矩形数组
	if (vEllipse.size() < 2)//不存在装甲矩形，直接返回
		return vRlt;
	for (unsigned int i = 0; i < vEllipse.size()-1;i++)//求任意两个旋转矩形的夹角
	{
		for (unsigned int j = i + 1; j < vEllipse.size(); j++)
		{
			dAngle = abs(vEllipse[i].angle - vEllipse[j].angle);
			while (dAngle > 180)
				dAngle -= 180;
			//判断这两个旋转矩形是否是一个装甲的两个LED灯条
			if ((dAngle < T_ANGLE_THRE || 180 - dAngle < T_ANGLE_THRE)
				&& abs(vEllipse[i].size.height - vEllipse[j].size.height) < (vEllipse[i].size.height + vEllipse[j].size.height) / T_SIZE_THRE
				&&abs(vEllipse[i].size.width - vEllipse[j].size.width) < (vEllipse[i].size.width + vEllipse[j].size.width) / T_SIZE_THRE)
			{
				armor.center.x = (vEllipse[i].center.x + vEllipse[j].center.x) / 2;//装甲中心x坐标
				armor.center.y = (vEllipse[i].center.y + vEllipse[j].center.y) / 2;//装甲中心y坐标
				armor.angle = (vEllipse[i].angle + vEllipse[j].angle) / 2;//装甲所在矩形的旋转角度
				if (180 - dAngle < T_ANGLE_THRE)
					armor.angle += 90;
				nL = (vEllipse[i].size.height + vEllipse[j].size.height) / 2;//装甲所在矩形的高度
				nW = sqrt((vEllipse[i].center.x - vEllipse[j].center.x)*(vEllipse[i].center.x - vEllipse[j].center.x)
					+ (vEllipse[i].center.y - vEllipse[j].center.y)*(vEllipse[i].center.y - vEllipse[j].center.y));//装甲的宽度等于两侧LED所在旋转矩形中心坐标的距离
				/*if (nL<nW)
				{*/
				armor.size.height = nL;
				armor.size.width = nW;
				/*}
				else {
					armor.size.height = nW;
					armor.size.width = nL;
				}*/
				Vec3b centerPixel = src.at<Vec3b>(armor.center.y, armor.center.x);
				if(centerPixel.val[0]>50||centerPixel.val[1]>50||centerPixel.val[2]>50)//装甲旋转矩形中间点不是黑色的
					continue;

				//漫水填充装甲矩形的中间局域
				Point2f ptI[4], ptJ[4];
				for (int i = 0; i < 4; i++)//初始化
				{
					ptI[i].x = 0;
					ptI[i].y = 0;
					ptJ[i].x = 0;
					ptJ[i].y = 0;
				}
				vEllipse[i].points(ptI);//获取旋转矩形的四个点坐标
				vEllipse[j].points(ptJ);
				int leftX = 0, rightX = 10000;
				if (vEllipse[i].center.x < vEllipse[j].center.x)//第i个矩形在左边，第j个矩形的右边
				{
					for (int i = 0; i < 4;i++)//找出感兴趣局域直方矩形的左右坐标
					{
						if (ptI[i].x > leftX)
							leftX = ptI[i].x;
						if (ptJ[i].x < rightX)
							rightX = ptJ[i].x;

					}
				}
				else //第j个矩形的左边，第i个矩形在右边
				{
					for (int i = 0; i < 4;i++)
					{
						if (ptJ[i].x > leftX)
							leftX = ptJ[i].x;
						if (ptI[i].x < rightX)
							rightX = ptI[i].x;
					}
				}
				int roiWidth = abs(rightX - leftX);
				int roiHight = (vEllipse[i].size.height+vEllipse[j].size.height)/4;
				int topY = armor.center.y - (roiHight / 2) < 0 ? 0 : armor.center.y - (roiHight / 2);//保证y坐标不小于零
				Point2f leftTop(leftX,topY);//感兴趣局域的左上角坐标
				if (topY + roiHight > src.rows - 1||leftX+roiWidth>src.cols-1)//感兴趣区域超出了原图范围
				{
					continue;
				}
				ROI_img = src(Rect(leftTop.x,leftTop.y,roiWidth,roiHight)).clone();
				Rect ccomp;
				//进行漫水填充
				if (ROI_img.rows*ROI_img.cols>0)
				  {
					int fillPoints = floodFill(ROI_img, Point(ROI_img.cols / 2, ROI_img.rows / 2),
						Scalar(0, 0, 255), &ccomp, Scalar(lowDifference, lowDifference, lowDifference),
						Scalar(upDifference, upDifference, upDifference));
					int allPoints = ROI_img.rows*ROI_img.cols;
					if (fillPoints > allPoints * 4 / 5)//旋转矩形中间区域颜色基本一致
					{
						vRlt.push_back(armor);//将找出的装甲的旋转矩阵保存到vector
					}
				}
				else
					vRlt.push_back(armor);
			}
		}
	}

	return vRlt;
}

//标记出装甲举证
void drawBox(RotatedRect box, Mat img)
{
	Point2f pt[4];
	for (int i = 0; i < 4;i++)//初始化
	{
		pt[i].x = 0;
		pt[i].y = 0;
	}
	box.points(pt);
	line(img, pt[0], pt[1], CV_RGB(0, 0, 255), 2);
	line(img, pt[1], pt[2], CV_RGB(0, 0, 255), 2);
	line(img, pt[2], pt[3], CV_RGB(0, 0, 255), 2);
	line(img, pt[3], pt[0], CV_RGB(0, 0, 255), 2);
}
//计算一个模板的颜色最小阈值和最大阈值，并返回平均值
Scalar findColorThre(Mat src, Scalar & minColorThre, Scalar & maxColorThre)
{
	Scalar average;
	int  pixelNum = src.rows*src.cols;
	if (pixelNum > 10000)
		throw"模板大于100x100,请使用更小的模板！";
	int  minR = 255, minB = 255, minG = 255;
	int  maxR = 0, maxB = 0, maxG = 0;
	double sumR = 0, sumB = 0, sumG = 0;
	for (int i = 0; i < src.rows;i++)
	{
		Vec3b* rowP = src.ptr<Vec3b>(i);
		for (int j = 0; j < src.cols;j++)
		{
			if (rowP[j][0] < minB)//找出最小的B值
				minB = rowP[j][0];
			if (rowP[j][0] > maxB)//找出最大的B值
				maxB = rowP[j][0];
			if (rowP[j][1] < minG)
				minG = rowP[j][1];
			if (rowP[j][1] > maxG)
				maxG = rowP[j][1];
			if (rowP[j][2] < minR)
				minR = rowP[j][2];
			if (rowP[j][2] > maxR)
				maxR = rowP[j][2];
			sumB += rowP[j][0];
			sumG += rowP[j][1];
			sumR += rowP[j][2];
		}
	}
	average = Scalar(sumB / pixelNum, sumG / pixelNum, sumR / pixelNum);
	minColorThre = Scalar(minB, minG, minR);
	maxColorThre = Scalar(maxB, maxG, maxR);
	return average;

}
