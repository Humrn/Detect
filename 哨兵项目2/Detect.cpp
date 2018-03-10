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

//��ʾ���ڲ����Ľ���
void  Detect::showAdjustUGI()
{
	Mat    logoImg = imread("logo.png");
	namedWindow(winName);
	//createTrackbar("Ŀ�공����ɫ", winName,, 3);
	createTrackbar("LED�ƽǶ���ֵ", winName, &T_ANGLE_THRE, 50);
	createTrackbar("LED�ƵĴ�С��ֵ", winName, &T_SIZE_THRE, 10);
	createTrackbar("LED�Ƶ�br��������ֵ", winName, &br, 150);
	createTrackbar("��ˮ���������ֵ", winName, &upDifference, 10);
	createTrackbar("��ˮ�ĸ������ֵ", winName, &lowDifference, 10);
	createTrackbar("���ȵ�����ֵ", winName, &brightThre, 210);
	createTrackbar("���ͺ˳ߴ�", winName, &dilateSize, 7);
	createTrackbar("��ʴ�˳ߴ�", winName, &erodeSize, 7);
	createTrackbar("�Ƿ���ʾ��ˮЧ��", winName, &isShowFloorFill, 1);
	imshow(winName, logoImg);

}
//�رյ��ڲ����Ľ���
void   Detect::closeAdjustUGI()
{
	destroyWindow(winName);
}

//��������,ͻ��LED��
void Detect::brightAdjust(Mat src, Mat dst, double dContrast, double dBright)
{
	int nVal;
	if (dst.empty())//���Ŀ��ͼ��û�з����ڴ棬������ڴ�
	{
		dst = Mat(src.size(), src.type());
	}
	//	//��8�������̣߳�ִ������forѭ���ӿ������ٶ�
	//	omp_set_num_threads(8);
	//#pragma omp parallel for
	for (int i = 0; i < src.rows; i++)
	{
		Vec3b* p1 = src.ptr<Vec3b>(i);//��ȡԭͼ����ָ��
		Vec3b* p2 = dst.ptr<Vec3b>(i);//��ȡĿ��ͼ����ָ��
									  //
		for (int j = 0; j < src.cols; j++)
		{

			for (int k = 0; k < 3; k++)
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
void Detect::getDiffImage(Mat srcR, Mat srcB, Mat dst, int nThre, DetectColor t_led_color)
{
	if (dst.empty())
	{
		throw"Ŀ��ͼ��û���ڴ�ռ䣬���ʼ����";
	}
	omp_set_num_threads(8);
#pragma omp parallel for
	for (int i = 0; i < srcR.rows; i++)
	{
		//��ȡͼ�����ָ��
		uchar * pchar1 = srcR.ptr<uchar>(i);
		uchar * pchar2 = srcB.ptr<uchar>(i);
		uchar * pchar3 = dst.ptr<uchar>(i);
		for (int j = 0; j < srcR.cols; j++)
		{
			switch (t_led_color)
			{
			case Red://Ҫ�����Ǻ쳵
				if (pchar1[j] - pchar2[j]>nThre)
				{
					pchar3[j] = 255;
				}
				else
				{
					pchar3[j] = 0;
				}
				break;
			case Blue://Ҫ����������
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

//��ֵͼ����
void Detect::optimizeBinary(Mat src, Mat& binaryImg, Mat& dst, int nSize)
{
	if (src.empty() || binaryImg.empty())
	{
		throw"error:ԭͼ�Ͷ�ֵͼ��û�г�ʼ������Ϊ�ڷ����ڴ�!";
		return;
	}
	if (src.rows != binaryImg.rows || src.cols != binaryImg.cols)
	{
		throw"error:ԭͼ�Ĵ�С�Ͷ�ֵͼ�Ĵ�С��һ�£�";
		return;
	}
	
    Mat tempImg = Mat(binaryImg.size(), binaryImg.type());
	int offer = nSize / 2;

	for (int i = 0; i < binaryImg.rows; i++)
	{
		//��ȡͼ�����ָ��
		Vec3b * pSrc = src.ptr<Vec3b>(i);
		uchar * pBinary = binaryImg.ptr<uchar>(i);
		uchar * pdst = tempImg.ptr<uchar>(i);
		for (int j = 0; j < binaryImg.cols; j++)
		{
			if (pBinary[j] > 250)//��ǰ��ֵͼ�����ص��ֵΪ255
			{
				bool isBabPoint = true;
				for (int nI = 0; nI < nSize; nI++)//��⵱ǰ���ص�nSize*nSize�����ڵ����ص��Ƿ񶼵���255
				{
					for (int nJ = 0; nJ < nSize; nJ++)
					{
						int x = j + nJ - offer;
						int y = i + nI - offer;
						if (x<0 || x>binaryImg.cols - 1 || y<0 || y>binaryImg.rows - 1)//������ͼ��Χ
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
			else {//��ֵ������Ŀ������
				pdst[j] = pBinary[j];
			}
		}
	}
	dst = tempImg;
}

//���װ��λ��
vector<RotatedRect> Detect::armorDetect(Mat src, vector<RotatedRect> vEllipse)
{
	vector<RotatedRect> vRlt;
	Mat   ROI_img;//��ת���θ���Ȥ����
	RotatedRect armor;//����װ�׾������ת����
	int nL, nW;
	double dAngle;
	vRlt.clear();//���װ�׾������ת��������
	if (vEllipse.size() < 2)//������װ�׾��Σ�ֱ�ӷ���
		return vRlt;
	for (unsigned int i = 0; i < vEllipse.size() - 1; i++)//������������ת���εļн�
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
																												   
				armor.size.height = nL;
				armor.size.width = nW;
				if (armor.size.width / armor.size.height>3||armor.size.height/armor.size.width>3.5)//��߲�����Ҫ������װ�����ڵ���ת����
				{
					continue;
				}
				if (armor.center.y<0 || armor.center.y>src.rows - 1 || armor.center.x<0 || armor.center.x>src.cols - 1)//���ĵ㲻��ͼ����
					continue;
				
				bool isAmorRect = true;
				for (int i = 0; i < 3;i++)//���װ����ת�����м���ǲ��Ǻ�ɫ��
				{
					for (int j = 0; j < 3; j++)
					{
						Vec3b centerPixel;
						if (armor.center.y - 1 + i >= 0 && armor.center.y - 1 + i < src.rows - 1
							&& armor.center.x - 1 + i >= 0 && armor.center.x - 1 + i < src.cols - 1)//����Խ������
						{
							centerPixel = src.at<Vec3b>(armor.center.y, armor.center.x);
						}
						else
							continue;
						if (centerPixel.val[0] > armorCenterColorThre.val[0] || centerPixel.val[1] > armorCenterColorThre.val[1] || centerPixel.val[2]>armorCenterColorThre.val[2])//����װ�׾���
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
					for (int i = 0; i < 4; i++)//�ҳ�����Ȥ����ֱ�����ε���������
					{
						if (ptI[i].x > leftX)
							leftX = ptI[i].x;
						if (ptJ[i].x < rightX)
							rightX = ptJ[i].x;

					}
				}
				else //��j�����ε���ߣ���i���������ұ�
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
				int topY = armor.center.y - (roiHight / 2) < 0 ? 0 : armor.center.y - (roiHight / 2);//��֤y���겻С����
				Point2f leftTop(leftX, topY);//����Ȥ��������Ͻ�����
				if (leftX<0||leftX>src.cols-1
					||topY<0||topY>src.rows-1
					||topY + roiHight > src.rows - 1 || leftX + roiWidth>src.cols - 1)//����Ȥ���򳬳���ԭͼ��Χ
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
				//������ˮ���
				if (ROI_img.rows*ROI_img.cols>0)
				{
					int fillPoints = floodFill(ROI_img, Point(ROI_img.cols / 2, ROI_img.rows / 2),
						Scalar(0, 0, 255), &ccomp, Scalar(lowDifference, lowDifference, lowDifference),
						Scalar(upDifference, upDifference, upDifference));
					int allPoints = ROI_img.rows*ROI_img.cols;
					if (fillPoints > allPoints * (floorFillThre-1) / floorFillThre)//��ת�����м�������ɫ����һ��
					{
						vRlt.push_back(armor);//���ҳ���װ�׵���ת���󱣴浽vector
					}
				}
			}
		}
	}

	return vRlt;
}

//��ǳ�װ�׾���   
void Detect::drawBox(RotatedRect box, Mat img)
{
	Point2f pt[4];
	for (int i = 0; i < 4; i++)//��ʼ��
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

//����һ��ģ�����ɫ��С��ֵ�������ֵ��������ƽ��ֵ
Scalar Detect::findColorThre(Mat src, Scalar & minColorThre, Scalar & maxColorThre)
{
	Scalar average;
	int  pixelNum = src.rows*src.cols;
	if (pixelNum > 10000)
		throw"ģ�����100x100,��ʹ�ø�С��ģ�壡";
	int  minR = 255, minB = 255, minG = 255;
	int  maxR = 0, maxB = 0, maxG = 0;
	double sumR = 0, sumB = 0, sumG = 0;
	for (int i = 0; i < src.rows; i++)
	{
		Vec3b* rowP = src.ptr<Vec3b>(i);
		for (int j = 0; j < src.cols; j++)
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

//̽��װ����ת�����ҳ�����Ŀ��
RotatedRect  Detect::getAttackTarget()//��ȡ����Ŀ��
{
	//ģ�崦��
	if (ledPath != "")
	{
		Mat LEDcenterImg = imread(ledPath);//����LED����������ģ��ͼ
		findColorThre(LEDcenterImg, minColorThre, maxColorThre);//����LED������������ɫ��ֵ
		cout << "minB��" << minColorThre.val[0] << "   minG:" << minColorThre.val[1] << "  minR:" << minColorThre.val[2] << endl;
		cout << "maxB��" << maxColorThre.val[0] << "   maxG:" << maxColorThre.val[1] << "  maxR:" << maxColorThre.val[2] << endl;
	}

	//�����ʶ���ͼ��
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
	myCamera >> srcImg;//��ȡһ��ͼƬ
	//��ͼƬ��С���ٴ�����ʡ����ʱ��
	resize(srcImg, resizeImg, frameSize, 0, 0, INTER_NEAREST);
	//���������
	resizeW = frameSize.width / srcImg.cols;
	resizeH = frameSize.height / srcImg.rows;
	//imshow("ԭͼ", srcImg);
	Size imgSize;
	RotatedRect s;//������ת����
	vector<RotatedRect> vEllipse;//������ת��������������ڴ洢���ֵ�Ŀ����������
	vector<RotatedRect> vRlt;//װ����ת��������
	//vector<RotatedRect> vArmor;
	bool bFlag = false;
	vector<vector<Point>> contour;//���ڴ洢��⵽������
	vector<Vec4i>         hierarchy;//���ڴ洢�����Ĳ����Ϣ
	imgSize = resizeImg.size();

	//����һЩ�������飬���ڴ���ͼ��
	Mat rawImg = Mat(imgSize, resizeImg.type());//�洢���ȵ������ͼƬ

	Mat grayImg = Mat(imgSize, CV_8UC1);
	Mat rImg = Mat(imgSize, CV_8UC1);
	Mat gImg = Mat(imgSize, CV_8UC1);
	Mat bImg = Mat(imgSize, CV_8UC1);
	Mat binaryImg = Mat(imgSize, CV_8UC1);
	Mat goodBinaryImg;
	Mat rltImg = Mat(imgSize, CV_8UC1);

	while (1)
	{
		if (myCamera.read(srcImg))//��ȡ����һ��ͼƬ
		{
			//��ͼƬ��С���ٴ�����ʡ����ʱ��
			resize(srcImg, resizeImg, frameSize, 0, 0, INTER_NEAREST);
			double beginTime = getTickCount();
			//blur(resizeImg, resizeImg, Size(3,3));//��ֵ�˲�
			brightAdjust(resizeImg, rawImg, 1, -brightThre);//����ͼƬ���ȣ�ͻ��LED��
			Mat bgr[3];
			split(rawImg, bgr);//������ͨ�������ط���
			bImg = bgr[0];
			gImg = bgr[1];
			rImg = bgr[2];
			//ͼ���ֵ�����������Rֵ��Gֵ��֮�����25���򷵻صĶ�ֵͼ���ֵΪ255������Ϊ0
		    getDiffImage(rImg, gImg, binaryImg, br,T_LED_COLOR);
			//imshow("��ֵͼ", binaryImg);
			//��ֵͼ�Ż�����
			optimizeBinary(resizeImg, binaryImg, binaryImg, 3);
			cv::imshow("�Ż���Ķ�ֵͼ", binaryImg);
			dilate(binaryImg, grayImg, Mat(), Point(-1, -1), 3);//ͼ������
			erode(grayImg, rltImg, Mat(), Point(-1, -1), 1);//ͼ��ʴ���������ٸ�ʴ���ڱ�����
			//�ٴ��Ż�
			optimizeBinary(resizeImg, rltImg, rltImg, 3);
			cv::imshow("������ͼ", rltImg);
			contour.clear();
			hierarchy.clear();
			findContours(rltImg, contour,hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);//�ڶ�ֵͼ����Ѱ������
			if(contour.size()<=0)//û���ҵ������������
				continue;
			for (int i = 0; i >= 0; i = hierarchy[i][0])//�������ж�������
			{
				if (contour[i].size() > 4)//�жϵ�ǰ�����Ƿ����5�����ص�
				{
					bFlag = true;//�������5�������⵽��Ŀ������
								 //���Ŀ�������Ϊ��Բ������һ��ѡ�����
					s = fitEllipse(contour[i]);
					if (ledPath != "")//ƥ����ε����ľ����Ƿ���LED��
					{
						for (int nI = 0; nI < 1; nI++)//��������ת�������ĵ�Ϊ���ĵ�1*1�����ؿ�
						{
							for (int nJ = 0; nJ < 1; nJ++)
							{
								int x = s.center.x - 0 + nJ;
								int y = s.center.y - 0 + nI;
								if (x > 0 && x < resizeImg.cols && y>0 && y < resizeImg.rows)
								{
									Vec3b v3b = resizeImg.at<Vec3b>(y, x);
									//����������򲻽ӽ�ģ����ɫ���಻��LED����
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
			//������⵽��LED����ת����
			for (int i = 0; i < vEllipse.size(); i++)
			{
				drawBox(vEllipse[i], rawImg);
			}
			//�����ӳ����������LED���ڵ���ת���ε�vector���ҳ�װ�׵�λ�ã�����װ����ת���Σ�����vector������
			vRlt = armorDetect(resizeImg, vEllipse);
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
			hierarchy.clear();
			contour.clear();
			//vArmor.clear();
			cout << "����ʱ�䣺" << (getTickCount() - beginTime) / getTickFrequency() * 1000 << "ms" << endl;

			//�����̰�������
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
		else {//û�пɻ�ȡ��ͼƬ��
			break;
		}
	}
	//���ع�����Ŀ��
	RotatedRect attackTarget;
	return attackTarget;
}