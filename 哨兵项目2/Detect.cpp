#include "Detect.h"
using namespace std;
using namespace cv;
Detect::Detect()
{
}

Detect::Detect(String videoPath)
{
	this->videoPath = videoPath;
}

Detect::~Detect()
{
}

//显示调节参数的界面
void  Detect::showAdjustUGI()
{
	Mat    logoImg = imread("logo.png");
	namedWindow(winName);
	//createTrackbar("目标车的颜色", winName,, 3);
	createTrackbar("LED灯角度阈值", winName, &T_ANGLE_THRE, 50);
	createTrackbar("LED灯的大小阈值", winName, &T_SIZE_THRE, 10);
	createTrackbar("LED灯的br分量差阈值", winName, &br, 150);
	createTrackbar("漫水的正差最大值", winName, &upDifference, 10);
	createTrackbar("漫水的负差最大值", winName, &lowDifference, 10);
	createTrackbar("亮度调节阈值", winName, &brightThre, 210);
	createTrackbar("膨胀核尺寸", winName, &dilateSize, 7);
	createTrackbar("腐蚀核尺寸", winName, &erodeSize, 7);
	createTrackbar("是否显示漫水效果", winName, &isShowFloorFill, 1);
	imshow(winName, logoImg);

}
//关闭调节参数的界面
void   Detect::closeAdjustUGI()
{
	destroyWindow(winName);
}

//调整亮度,突出LED灯
void Detect::brightAdjust(Mat src, Mat dst, double dContrast, double dBright)
{
	int nVal;
	if (dst.empty())//如果目标图还没有分配内存，则分配内存
	{
		dst = Mat(src.size(), src.type());
	}
	//	//开8个并行线程，执行下面for循环加快运行速度
	//	omp_set_num_threads(8);
	//#pragma omp parallel for
	for (int i = 0; i < src.rows; i++)
	{
		Vec3b* p1 = src.ptr<Vec3b>(i);//获取原图的行指针
		Vec3b* p2 = dst.ptr<Vec3b>(i);//获取目标图的行指针
									  //
		for (int j = 0; j < src.cols; j++)
		{

			for (int k = 0; k < 3; k++)
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
void Detect::getDiffImage(Mat srcR, Mat srcB, Mat dst, int nThre, DetectColor t_led_color)
{
	if (dst.empty())
	{
		throw"目标图像没有内存空间，请初始化！";
	}
	omp_set_num_threads(8);
#pragma omp parallel for
	for (int i = 0; i < srcR.rows; i++)
	{
		//获取图像的行指针
		uchar * pchar1 = srcR.ptr<uchar>(i);
		uchar * pchar2 = srcB.ptr<uchar>(i);
		uchar * pchar3 = dst.ptr<uchar>(i);
		for (int j = 0; j < srcR.cols; j++)
		{
			switch (t_led_color)
			{
			case Red://要检测的是红车
				if (pchar1[j] - pchar2[j]>nThre)
				{
					pchar3[j] = 255;
				}
				else
				{
					pchar3[j] = 0;
				}
				break;
			case Blue://要检测的是蓝车
				if (pchar2[j] - pchar1[j]>nThre)
				{
					pchar3[j] = 255;
				}
				else
				{
					pchar3[j] = 0;
				}
				break;
			case RedAndBlue:
				if (abs(pchar1[j] - pchar2[j])>nThre)
				{
					pchar3[j] = 255;
				}
				else
				{
					pchar3[j] = 0;
				}
				break;
			}
		}
	}
}

//二值图像降噪
void Detect::optimizeBinary(Mat src, Mat& binaryImg, Mat& dst, int nSize)
{
	if (src.empty() || binaryImg.empty())
	{
		throw"error:原图和二值图还没有初始化，请为期分配内存!";
		return;
	}
	if (src.rows != binaryImg.rows || src.cols != binaryImg.cols)
	{
		throw"error:原图的大小和二值图的大小不一致！";
		return;
	}
	
    Mat tempImg = Mat(binaryImg.size(), binaryImg.type());
	int offer = nSize / 2;

	for (int i = 0; i < binaryImg.rows; i++)
	{
		//获取图像的行指针
		Vec3b * pSrc = src.ptr<Vec3b>(i);
		uchar * pBinary = binaryImg.ptr<uchar>(i);
		uchar * pdst = tempImg.ptr<uchar>(i);
		for (int j = 0; j < binaryImg.cols; j++)
		{
			if (pBinary[j] > 250)//当前二值图的像素点的值为255
			{
				bool isBabPoint = true;
				for (int nI = 0; nI < nSize; nI++)//检测当前像素点nSize*nSize邻域内的像素点是否都等于255
				{
					for (int nJ = 0; nJ < nSize; nJ++)
					{
						int x = j + nJ - offer;
						int y = i + nI - offer;
						if (x<0 || x>binaryImg.cols - 1 || y<0 || y>binaryImg.rows - 1)//超出了图像范围
						{
							continue;
						}
						else {
							uchar ch_temp = binaryImg.at<char>(y, x);
							if (ch_temp < 250)
							{
								isBabPoint = false;
								break;
							}
						}

					}
					if (isBabPoint == false)
						break;
				}
				if (isBabPoint)
				{
					pdst[j] = 0;
				}
				else {
					pdst[j] = pBinary[j];
				}
			}
			else {//将值拷贝到目标数组
				pdst[j] = pBinary[j];
			}
		}
	}
	dst = tempImg;
}

//检测装甲位置
vector<RotatedRect> Detect::armorDetect(Mat src, vector<RotatedRect> vEllipse)
{
	vector<RotatedRect> vRlt;
	Mat   ROI_img;//旋转矩形感兴趣局域
	RotatedRect armor;//定义装甲局域的旋转矩形
	int nL, nW;
	double dAngle;
	vRlt.clear();//清空装甲局域的旋转矩形数组
	if (vEllipse.size() < 2)//不存在装甲矩形，直接返回
		return vRlt;
	for (unsigned int i = 0; i < vEllipse.size() - 1; i++)//求任意两个旋转矩形的夹角
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
																												   
				armor.size.height = nL;
				armor.size.width = nW;
				if (armor.size.width / armor.size.height>3||armor.size.height/armor.size.width>3.5)//宽高不符合要求，则不是装甲所在的旋转矩形
				{
					continue;
				}
				if (armor.center.y<0 || armor.center.y>src.rows - 1 || armor.center.x<0 || armor.center.x>src.cols - 1)//中心点不在图像中
					continue;
				
				bool isAmorRect = true;
				for (int i = 0; i < 3;i++)//检测装甲旋转矩形中间点是不是黑色的
				{
					for (int j = 0; j < 3; j++)
					{
						Vec3b centerPixel;
						if (armor.center.y - 1 + i >= 0 && armor.center.y - 1 + i < src.rows - 1
							&& armor.center.x - 1 + i >= 0 && armor.center.x - 1 + i < src.cols - 1)//访问越界限制
						{
							centerPixel = src.at<Vec3b>(armor.center.y, armor.center.x);
						}
						else
							continue;
						if (centerPixel.val[0] > armorCenterColorThre.val[0] || centerPixel.val[1] > armorCenterColorThre.val[1] || centerPixel.val[2]>armorCenterColorThre.val[2])//不是装甲矩阵
						{
							isAmorRect = false;
							break;
						}

					}
					if (isAmorRect==false)
					{
						break;
					}
				}
				if (isAmorRect==false)
				{
					continue;
				}
				

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
					for (int i = 0; i < 4; i++)//找出感兴趣局域直方矩形的左右坐标
					{
						if (ptI[i].x > leftX)
							leftX = ptI[i].x;
						if (ptJ[i].x < rightX)
							rightX = ptJ[i].x;

					}
				}
				else //第j个矩形的左边，第i个矩形在右边
				{
					for (int i = 0; i < 4; i++)
					{
						if (ptJ[i].x > leftX)
							leftX = ptJ[i].x;
						if (ptI[i].x < rightX)
							rightX = ptI[i].x;
					}
				}
				int roiWidth = abs(rightX - leftX);
				int roiHight = 5;
				int topY = armor.center.y - (roiHight / 2) < 0 ? 0 : armor.center.y - (roiHight / 2);//保证y坐标不小于零
				Point2f leftTop(leftX, topY);//感兴趣局域的左上角坐标
				if (leftX<0||leftX>src.cols-1
					||topY<0||topY>src.rows-1
					||topY + roiHight > src.rows - 1 || leftX + roiWidth>src.cols - 1)//感兴趣区域超出了原图范围
				{
					continue;
				}
				if (isShowFloorFill==1)
				{
					ROI_img = src(Rect(leftTop.x, leftTop.y, roiWidth, roiHight));
				}
				else {
					ROI_img = src(Rect(leftTop.x, leftTop.y, roiWidth, roiHight)).clone();
				}
				Rect ccomp;
				//进行漫水填充
				if (ROI_img.rows*ROI_img.cols>0)
				{
					int fillPoints = floodFill(ROI_img, Point(ROI_img.cols / 2, ROI_img.rows / 2),
						Scalar(0, 0, 255), &ccomp, Scalar(lowDifference, lowDifference, lowDifference),
						Scalar(upDifference, upDifference, upDifference));
					int allPoints = ROI_img.rows*ROI_img.cols;
					if (fillPoints > allPoints * (floorFillThre-1) / floorFillThre)//旋转矩形中间区域颜色基本一致
					{
						vRlt.push_back(armor);//将找出的装甲的旋转矩阵保存到vector
					}
				}
			}
		}
	}

	return vRlt;
}

//标记出装甲矩阵   
void Detect::drawBox(RotatedRect box, Mat img)
{
	Point2f pt[4];
	for (int i = 0; i < 4; i++)//初始化
	{
		pt[i].x = 0;
		pt[i].y = 0;
	}
	box.points(pt);
	line(img, pt[0], pt[1], CV_RGB(0, 255 ,0), 2);
	line(img, pt[1], pt[2], CV_RGB(0, 255 ,0), 2);
	line(img, pt[2], pt[3], CV_RGB(0, 255, 0), 2);
	line(img, pt[3], pt[0], CV_RGB(0, 255, 0), 2);
}

//计算一个模板的颜色最小阈值和最大阈值，并返回平均值
Scalar Detect::findColorThre(Mat src, Scalar & minColorThre, Scalar & maxColorThre)
{
	Scalar average;
	int  pixelNum = src.rows*src.cols;
	if (pixelNum > 10000)
		throw"模板大于100x100,请使用更小的模板！";
	int  minR = 255, minB = 255, minG = 255;
	int  maxR = 0, maxB = 0, maxG = 0;
	double sumR = 0, sumB = 0, sumG = 0;
	for (int i = 0; i < src.rows; i++)
	{
		Vec3b* rowP = src.ptr<Vec3b>(i);
		for (int j = 0; j < src.cols; j++)
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

//探测装甲旋转矩阵，找出攻击目标
RotatedRect  Detect::getAttackTarget()//获取攻击目标
{
	//模板处理
	if (ledPath != "")
	{
		Mat LEDcenterImg = imread(ledPath);//读入LED灯中心区域模板图
		findColorThre(LEDcenterImg, minColorThre, maxColorThre);//计算LED灯中心区域颜色阈值
		cout << "minB：" << minColorThre.val[0] << "   minG:" << minColorThre.val[1] << "  minR:" << minColorThre.val[2] << endl;
		cout << "maxB：" << maxColorThre.val[0] << "   maxG:" << maxColorThre.val[1] << "  maxR:" << maxColorThre.val[2] << endl;
	}

	//读入待识别的图像
	VideoCapture  myCamera;
	if(videoPath!="")
	{
		myCamera=VideoCapture(videoPath);
	}
	else
	{
		myCamera = VideoCapture(0);
		myCamera.set(CV_CAP_PROP_FRAME_WIDTH, frameSize.width);
		myCamera.set(CV_CAP_PROP_FRAME_HEIGHT, frameSize.height);
	}
	
	Mat resizeImg, srcImg;
	//Mat srcImg = imread("chezi.png");
	myCamera >> srcImg;//获取一张图片
	//将图片缩小后再处理，节省处理时间
	resize(srcImg, resizeImg, frameSize, 0, 0, INTER_NEAREST);
	//计算放缩比
	resizeW = frameSize.width / srcImg.cols;
	resizeH = frameSize.height / srcImg.rows;
	//imshow("原图", srcImg);
	Size imgSize;
	RotatedRect s;//定义旋转矩形
	vector<RotatedRect> vEllipse;//定于旋转矩阵的向量，用于存储发现的目标区域轮廓
	vector<RotatedRect> vRlt;//装甲旋转矩形向量
	//vector<RotatedRect> vArmor;
	bool bFlag = false;
	vector<vector<Point>> contour;//用于存储检测到的轮廓
	vector<Vec4i>         hierarchy;//用于存储轮廓的层次信息
	imgSize = resizeImg.size();

	//定义一些缓冲数组，用于处理图像
	Mat rawImg = Mat(imgSize, resizeImg.type());//存储亮度调整后的图片

	Mat grayImg = Mat(imgSize, CV_8UC1);
	Mat rImg = Mat(imgSize, CV_8UC1);
	Mat gImg = Mat(imgSize, CV_8UC1);
	Mat bImg = Mat(imgSize, CV_8UC1);
	Mat binaryImg = Mat(imgSize, CV_8UC1);
	Mat goodBinaryImg;
	Mat rltImg = Mat(imgSize, CV_8UC1);

	while (1)
	{
		if (myCamera.read(srcImg))//获取到了一张图片
		{
			//将图片缩小后再处理，节省处理时间
			resize(srcImg, resizeImg, frameSize, 0, 0, INTER_NEAREST);
			double beginTime = getTickCount();
			//blur(resizeImg, resizeImg, Size(3,3));//均值滤波
			brightAdjust(resizeImg, rawImg, 1, -brightThre);//调整图片亮度，突出LED灯
			Mat bgr[3];
			split(rawImg, bgr);//将三个通道的像素分离
			bImg = bgr[0];
			gImg = bgr[1];
			rImg = bgr[2];
			//图像二值化，如果像素R值与G值得之差大于25，则返回的二值图像的值为255，否则为0
		    getDiffImage(rImg, gImg, binaryImg, br,T_LED_COLOR);
			//imshow("二值图", binaryImg);
			//二值图优化减噪
			optimizeBinary(resizeImg, binaryImg, binaryImg, 3);
			cv::imshow("优化后的二值图", binaryImg);
			dilate(binaryImg, grayImg, Mat(), Point(-1, -1), 3);//图像膨胀
			erode(grayImg, rltImg, Mat(), Point(-1, -1), 1);//图像腐蚀，先膨胀再腐蚀属于闭运算
			//再次优化
			optimizeBinary(resizeImg, rltImg, rltImg, 3);
			cv::imshow("闭运算图", rltImg);
			contour.clear();
			hierarchy.clear();
			findContours(rltImg, contour,hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);//在二值图像中寻找轮廓
			if(contour.size()<=0)//没有找到轮廓，则继续
				continue;
			for (int i = 0; i >= 0; i = hierarchy[i][0])//遍历所有顶层轮廓
			{
				if (contour[i].size() > 4)//判断当前轮廓是否大于5个像素点
				{
					bFlag = true;//如果大于5个，则检测到了目标区域
								 //拟合目标区域成为椭圆，返回一个选择矩形
					s = fitEllipse(contour[i]);
					if (ledPath != "")//匹配矩形的中心局域是否是LED灯
					{
						for (int nI = 0; nI < 1; nI++)//遍历以旋转矩形中心点为中心的1*1的像素块
						{
							for (int nJ = 0; nJ < 1; nJ++)
							{
								int x = s.center.x - 0 + nJ;
								int y = s.center.y - 0 + nI;
								if (x > 0 && x < resizeImg.cols && y>0 && y < resizeImg.rows)
								{
									Vec3b v3b = resizeImg.at<Vec3b>(y, x);
									//如果中心区域不接近模板颜色，侧不是LED轮廓
									if (v3b[0] <minColorThre.val[0] - 30 || v3b[1] < minColorThre.val[1] - 30 || v3b[2] < minColorThre.val[2] - 30)
										bFlag = false;
									if (v3b[0] > maxColorThre.val[0] + 30 || v3b[1] > maxColorThre.val[1] + 30 || v3b[2]>maxColorThre.val[2] + 30)
										bFlag = false;
								}
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
			for (int i = 0; i < vEllipse.size(); i++)
			{
				drawBox(vEllipse[i], rawImg);
			}
			//调用子程序，在输入的LED所在的旋转矩形的vector中找出装甲的位置，并包装成旋转矩形，存入vector并返回
			vRlt = armorDetect(resizeImg, vEllipse);
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
			hierarchy.clear();
			contour.clear();
			//vArmor.clear();
			cout << "处理时间：" << (getTickCount() - beginTime) / getTickFrequency() * 1000 << "ms" << endl;

			//检测键盘按键输入
			char pressedKey = waitKey(1);
			if (pressedKey == 27)
				break;
			if (pressedKey == '1')
			{
				continue;
			}
			if (pressedKey=='2')
			{
				showAdjustUGI();
			}
			if (pressedKey=='3')
			{
				closeAdjustUGI();
			}
			if (pressedKey=='4')
			{
				myCamera.set(CV_CAP_PROP_POS_FRAMES, 0);
			}
		}
		else {//没有可获取的图片了
			break;
		}
	}
	//返回攻击的目标
	RotatedRect attackTarget;
	return attackTarget;
}