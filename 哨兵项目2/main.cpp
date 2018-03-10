#include<iostream>
#include"opencv2/core.hpp"
#include"opencv2/highgui.hpp"
#include"opencv2/video.hpp"
#include"opencv2/imgproc.hpp"
#include "omp.h"
#include<time.h>


using namespace std;
using namespace cv;

//�궨��
#define T_ANGLE_THRE 15   //�Ƕ���ֵ
#define T_SIZE_THRE  5     //�ߴ���ֵ
#define T_LED_COLOR  2    //����Ҫʶ���װ�׵���ɫ��1������ɫ��2�����ɫ,3����ʶ��

//��������
void                brightAdjust(Mat src, Mat dst, double dContrast, double dBright);//���ȵ��ں���
void                getDiffImage(Mat src1, Mat src2, Mat dst, int nThre);//��ֵ��
vector<RotatedRect> armorDetect(Mat src, vector<RotatedRect> vEllipse);//���װ��λ��
void                drawBox(RotatedRect box, Mat img);//��ǳ�װ�׾���
Scalar              findColorThre(Mat src, Scalar & minColorThre, Scalar & maxColorThre);//����һ��ģ�����ɫ��С��ֵ�������ֵ��������ƽ��ֵ
//ȫ�ֱ�������
double resizeW = 0.4;//ͼ���ķ�������,Ĭ��Ϊ0.4
double resizeH = 0.5;//ͼ��ߵķ���������Ĭ��Ϊ0.5
Size frameSize=Size(800,600);//��ȡͼƬ�Ĵ�С
Scalar minColorThre=Scalar(20,20,20);//Ĭ��LED����ɫ��Χ
Scalar maxColorThre=Scalar(250,255,255);
int lowDifference = 3;//��ˮ���ĸ������ֵ���������ֵ
int upDifference = 3;
int brightThre = -120;//���ȵ�����ֵ
int gr = 60;

void main1()
{
	////ģ�崦��
	//Mat LEDcenterImg = imread("yellow4.jpg");//����LED����������ģ��ͼ
	//findColorThre(LEDcenterImg, minColorThre, maxColorThre);//����LED������������ɫ��ֵ
	//cout << "minB��" << minColorThre.val[0] << "   minG:" << minColorThre.val[1] << "  minR:" << minColorThre.val[2] << endl;
	//cout << "maxB��" << maxColorThre.val[0] << "   maxG:" << maxColorThre.val[1] << "  maxR:" << maxColorThre.val[2] << endl;
	//
	
	//����ͼ��ʶ��Ŀ��
	VideoCapture myCamera("�����زĺ쳵��ת-ev-0.MOV");
	myCamera.set(CV_CAP_PROP_FRAME_WIDTH, 300);
	myCamera.set(CV_CAP_PROP_FRAME_HEIGHT, 300);
    Mat resizeImg,srcImg;
	//Mat srcImg = imread("chezi.png");
	myCamera >> srcImg;//��ȡһ��ͼƬ
    //��ͼƬ��С���ٴ�����ʡ����ʱ��
	resize(srcImg, resizeImg,frameSize,0,0,INTER_NEAREST);
	//���������
	resizeW = frameSize.width / srcImg.cols;
	resizeH = frameSize.height / srcImg.rows;
    //imshow("ԭͼ", srcImg);
	Size imgSize;
	RotatedRect s;//������ת����
	vector<RotatedRect> vEllipse;//������ת��������������ڴ洢���ֵ�Ŀ����������
	vector<RotatedRect> vRlt;//װ����ת��������
	vector<RotatedRect> vArmor;
	bool bFlag = false;
	vector<vector<Point>> contour;//���ڴ洢��⵽������
	imgSize = resizeImg.size();

	//����һЩ�������飬���ڴ���ͼ��
	Mat rawImg = Mat(imgSize, resizeImg.type());//�洢���ȵ������ͼƬ
	
	Mat grayImg = Mat(imgSize, CV_8UC1);
	Mat rImg = Mat(imgSize, CV_8UC1);
	Mat gImg = Mat(imgSize, CV_8UC1);
	Mat bImg = Mat(imgSize, CV_8UC1);
	Mat binaryImg = Mat(imgSize, CV_8UC1);
	Mat rltImg = Mat(imgSize, CV_8UC1); 
	
	while (1)
	{
		if (myCamera.read(srcImg))//��ȡ����һ��ͼƬ
		{
			//��ͼƬ��С���ٴ�����ʡ����ʱ��
			resize(srcImg, resizeImg,frameSize, 0, 0, INTER_NEAREST);
			double beginTime = getTickCount();
			vector<Mat> bgr;
			brightAdjust(resizeImg, rawImg, 1, brightThre);//����ͼƬ���ȣ�ͻ��LED��
			split(rawImg, bgr);//������ͨ�������ط���
			bImg = bgr[0];
			gImg = bgr[1];
			rImg = bgr[2];
			//�������Rֵ��Gֵ��֮�����25���򷵻صĶ�ֵͼ���ֵΪ255������Ϊ0
			if (T_LED_COLOR == 1)//Ŀ��Ϊ��ɫ
			{
				getDiffImage(bImg, rImg, binaryImg, gr);
			}
			else {//Ŀ��Ϊ��ɫ
				getDiffImage(rImg, bImg, binaryImg, gr);
			}
			imshow("��ֵͼ", binaryImg);
			dilate(binaryImg, grayImg, Mat(), Point(-1, -1), 3);//ͼ������
			erode(grayImg, rltImg, Mat(), Point(-1, -1), 1);//ͼ��ʴ���������ٸ�ʴ���ڱ�����
			imshow("������ͼ", rltImg);
			contour.clear();
			findContours(rltImg, contour, RETR_CCOMP, CHAIN_APPROX_SIMPLE);//�ڶ�ֵͼ����Ѱ������
			for (int i = 0; i < contour.size(); i++)
			{
				if (contour[i].size()>4)//�жϵ�ǰ�����Ƿ����10�����ص�
				{
					bFlag = true;//�������10�������⵽��Ŀ������
					//���Ŀ�������Ϊ��Բ������һ��ѡ�����
					s = fitEllipse(contour[i]);
					for (int nI = 0; nI < 1; nI++)//��������ת�������ĵ�Ϊ���ĵ�1*1�����ؿ�
					{
						for (int nJ = 0; nJ < 1; nJ++)
						{
							int x = s.center.x - 0 + nJ;
							int y = s.center.y - 0 + nI;
							if (x > 0 && x < resizeImg.cols&&y>0 && y < resizeImg.rows)
							{
								Vec3b v3b = resizeImg.at<Vec3b>(y, x);
								//����������򲻽ӽ�ģ����ɫ���಻��LED����
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
			//������⵽��LED����ת����
			for (int i = 0; i < vEllipse.size();i++)
			{
				drawBox(vEllipse[i], rawImg);
			}
			//�����ӳ����������LED���ڵ���ת���ε�vector���ҳ�װ�׵�λ�ã�����װ����ת���Σ�����vector������
			vRlt = armorDetect(resizeImg,vEllipse);
			for (unsigned int i = 0; i < vRlt.size(); i++)
			{
				drawBox(vRlt[i], resizeImg);
			}
			imshow("���ȵ���Ч��ͼ", rawImg);
			imshow("��С���Ч��ͼ", resizeImg);
			////���ԭͼ��װ�����ڵľ���
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
			//imshow("Ч��ͼ", srcImg);

			//�������
			vEllipse.clear();
			vRlt.clear();
			vArmor.clear();
			cout << "����ʱ�䣺" << (getTickCount() - beginTime) / getTickFrequency() * 1000 << "ms" << endl;
			char pressedKey = waitKey(1);
			if (pressedKey == 27)
				break;
			if (pressedKey=='1')
			{
				continue;
			}
		}
		else {//û�пɻ�ȡ��ͼƬ��
			break;
		}
	}

	waitKey(0);
	system("pause");
}

//��������,ͻ��LED��
void brightAdjust(Mat src, Mat dst, double dContrast, double dBright)
{
	int nVal;
	if (dst.empty())//���Ŀ��ͼ��û�з����ڴ棬������ڴ�
	{
		dst = Mat(src.size(), src.type());
	}
//	//��8�������̣߳�ִ������forѭ���ӿ������ٶ�
//	omp_set_num_threads(8);
//#pragma omp parallel for

	for (int i = 0; i < src.rows;i++)
	{
		Vec3b* p1 = src.ptr<Vec3b>(i);//��ȡԭͼ����ָ��
		Vec3b* p2 = dst.ptr<Vec3b>(i);//��ȡĿ��ͼ����ָ��
//
		for (int j = 0; j < src.cols; j++)
		{

			for (int k = 0; k < 3;k++)
			{
//#pragma omp critical
				//��ÿ�����ص�ÿ��ͨ����ֵ���������Ա任
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

//ͼ���ֵ����Ϊ���������׼��
void getDiffImage(Mat src1, Mat src2, Mat dst, int nThre)
{
	if (dst.empty())
	{
		throw"Ŀ��ͼ��û���ڴ�ռ䣬���ʼ����";
	}
	omp_set_num_threads(8);
#pragma omp parallel for
	for (int i = 0; i < src1.rows;i++)
	{
		//��ȡͼ�����ָ��
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

//���װ��λ��
vector<RotatedRect> armorDetect(Mat src,vector<RotatedRect> vEllipse)
{
	vector<RotatedRect> vRlt;
	Mat   ROI_img;//��ת���θ���Ȥ����
	RotatedRect armor;//����װ�׾������ת����
	int nL, nW;
	double dAngle;
	vRlt.clear();//���װ�׾������ת��������
	if (vEllipse.size() < 2)//������װ�׾��Σ�ֱ�ӷ���
		return vRlt;
	for (unsigned int i = 0; i < vEllipse.size()-1;i++)//������������ת���εļн�
	{
		for (unsigned int j = i + 1; j < vEllipse.size(); j++)
		{
			dAngle = abs(vEllipse[i].angle - vEllipse[j].angle);
			while (dAngle > 180)
				dAngle -= 180;
			//�ж���������ת�����Ƿ���һ��װ�׵�����LED����
			if ((dAngle < T_ANGLE_THRE || 180 - dAngle < T_ANGLE_THRE)
				&& abs(vEllipse[i].size.height - vEllipse[j].size.height) < (vEllipse[i].size.height + vEllipse[j].size.height) / T_SIZE_THRE
				&&abs(vEllipse[i].size.width - vEllipse[j].size.width) < (vEllipse[i].size.width + vEllipse[j].size.width) / T_SIZE_THRE)
			{
				armor.center.x = (vEllipse[i].center.x + vEllipse[j].center.x) / 2;//װ������x����
				armor.center.y = (vEllipse[i].center.y + vEllipse[j].center.y) / 2;//װ������y����
				armor.angle = (vEllipse[i].angle + vEllipse[j].angle) / 2;//װ�����ھ��ε���ת�Ƕ�
				if (180 - dAngle < T_ANGLE_THRE)
					armor.angle += 90;
				nL = (vEllipse[i].size.height + vEllipse[j].size.height) / 2;//װ�����ھ��εĸ߶�
				nW = sqrt((vEllipse[i].center.x - vEllipse[j].center.x)*(vEllipse[i].center.x - vEllipse[j].center.x)
					+ (vEllipse[i].center.y - vEllipse[j].center.y)*(vEllipse[i].center.y - vEllipse[j].center.y));//װ�׵Ŀ�ȵ�������LED������ת������������ľ���
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
				if(centerPixel.val[0]>50||centerPixel.val[1]>50||centerPixel.val[2]>50)//װ����ת�����м�㲻�Ǻ�ɫ��
					continue;

				//��ˮ���װ�׾��ε��м����
				Point2f ptI[4], ptJ[4];
				for (int i = 0; i < 4; i++)//��ʼ��
				{
					ptI[i].x = 0;
					ptI[i].y = 0;
					ptJ[i].x = 0;
					ptJ[i].y = 0;
				}
				vEllipse[i].points(ptI);//��ȡ��ת���ε��ĸ�������
				vEllipse[j].points(ptJ);
				int leftX = 0, rightX = 10000;
				if (vEllipse[i].center.x < vEllipse[j].center.x)//��i����������ߣ���j�����ε��ұ�
				{
					for (int i = 0; i < 4;i++)//�ҳ�����Ȥ����ֱ�����ε���������
					{
						if (ptI[i].x > leftX)
							leftX = ptI[i].x;
						if (ptJ[i].x < rightX)
							rightX = ptJ[i].x;

					}
				}
				else //��j�����ε���ߣ���i���������ұ�
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
				int topY = armor.center.y - (roiHight / 2) < 0 ? 0 : armor.center.y - (roiHight / 2);//��֤y���겻С����
				Point2f leftTop(leftX,topY);//����Ȥ��������Ͻ�����
				if (topY + roiHight > src.rows - 1||leftX+roiWidth>src.cols-1)//����Ȥ���򳬳���ԭͼ��Χ
				{
					continue;
				}
				ROI_img = src(Rect(leftTop.x,leftTop.y,roiWidth,roiHight)).clone();
				Rect ccomp;
				//������ˮ���
				if (ROI_img.rows*ROI_img.cols>0)
				  {
					int fillPoints = floodFill(ROI_img, Point(ROI_img.cols / 2, ROI_img.rows / 2),
						Scalar(0, 0, 255), &ccomp, Scalar(lowDifference, lowDifference, lowDifference),
						Scalar(upDifference, upDifference, upDifference));
					int allPoints = ROI_img.rows*ROI_img.cols;
					if (fillPoints > allPoints * 4 / 5)//��ת�����м�������ɫ����һ��
					{
						vRlt.push_back(armor);//���ҳ���װ�׵���ת���󱣴浽vector
					}
				}
				else
					vRlt.push_back(armor);
			}
		}
	}

	return vRlt;
}

//��ǳ�װ�׾�֤
void drawBox(RotatedRect box, Mat img)
{
	Point2f pt[4];
	for (int i = 0; i < 4;i++)//��ʼ��
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
//����һ��ģ�����ɫ��С��ֵ�������ֵ��������ƽ��ֵ
Scalar findColorThre(Mat src, Scalar & minColorThre, Scalar & maxColorThre)
{
	Scalar average;
	int  pixelNum = src.rows*src.cols;
	if (pixelNum > 10000)
		throw"ģ�����100x100,��ʹ�ø�С��ģ�壡";
	int  minR = 255, minB = 255, minG = 255;
	int  maxR = 0, maxB = 0, maxG = 0;
	double sumR = 0, sumB = 0, sumG = 0;
	for (int i = 0; i < src.rows;i++)
	{
		Vec3b* rowP = src.ptr<Vec3b>(i);
		for (int j = 0; j < src.cols;j++)
		{
			if (rowP[j][0] < minB)//�ҳ���С��Bֵ
				minB = rowP[j][0];
			if (rowP[j][0] > maxB)//�ҳ�����Bֵ
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
