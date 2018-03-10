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

////�궨��
//#define T_ANGLE_THRE 15   //�Ƕ���ֵ������LED�ƵĽǶȲ�ľ���ֵ����������ֵ
//#define T_SIZE_THRE  5    //�ߴ���ֵ,ֵԽ����ʶ���������Խ��
//#define T_LED_COLOR  2    //����Ҫʶ���װ�׵���ɫ��1������ɫ��2�����ɫ,3����ʶ��


//Ҫʶ���С����LED����ɫ
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
	//��������
	void                showAdjustUGI();//��ʾ�������ڽ���
	void                closeAdjustUGI();//�رղ������ڽ���
	void                optimizeBinary(Mat src, Mat& binaryImg, Mat& dst, int nSize);//�Ż���ֵͼ������ģ����Ե��ɵĸ���
	void                brightAdjust(Mat src, Mat dst, double dContrast, double dBright);//���ȵ��ں���
	void                getDiffImage(Mat src1, Mat src2, Mat dst, int nThre,DetectColor t_led_color=Red);//��ֵ��
	vector<RotatedRect> armorDetect(Mat src, vector<RotatedRect> vEllipse);//���װ��λ��
	void                drawBox(RotatedRect box, Mat img);//��ǳ�װ�׾���
	Scalar              findColorThre(Mat src, Scalar & minColorThre, Scalar & maxColorThre);//����һ��ģ�����ɫ��С��ֵ�������ֵ��������ƽ��ֵ
	RotatedRect         getAttackTarget();//��ȡ����Ŀ��
public:
	int                 T_ANGLE_THRE = 15; //�Ƕ���ֵ������LED�ƵĽǶȲ�ľ���ֵ����������ֵ
	int                 T_SIZE_THRE = 4; //�ߴ���ֵ,ֵԽ����ʶ���������Խ��
	int                 lowDifference = 4;//��ˮ���ĸ������ֵ���������ֵ��Ĭ�϶���3
	int                 upDifference = 3;
	int                 brightThre = 120;//���ȵ�����ֵ
	int                 br = 25;//LED�ı任��gֵ��rֵ֮����ֵ
	int                 dilateSize = 3;//���͵ĺ˴�С
	int                 erodeSize = 1;//��ʴ�ĺʹ�С
	int                 isShowFloorFill = 0;//�Ƿ���ʾ��ˮЧ��
	int                 floorFillThre = 5;//ͨ����ˮ������ֵ��ֵԽС��ͨ���Ļ���Խ����ʶ������ľ���Խ��
	String              videoPath="";//����ͼ���·������Ϊ�գ���ͼƬ��ԴΪ�豸������ͷ
	String              ledPath = "";//�Ƶ�ģ���·������Ϊ�գ�������LED��ģ��
	DetectColor         T_LED_COLOR = Red;//����Ҫʶ���װ�׵���ɫ,Ĭ��Ϊ��ɫ
	Scalar              armorCenterColorThre=Scalar(70,70,70);//װ�׾���������������ɫ��ֵ��Ĭ��Ϊrgbֵ��С��50
	Size                frameSize = Size(800, 600);//��ȡͼƬ�Ĵ�С
private:
	String              winName = "�������ڽ���";
	double              resizeW = 0.4;//ͼ���ķ�������,Ĭ��Ϊ0.4
	double              resizeH = 0.5;//ͼ��ߵķ���������Ĭ��Ϊ0.5
	Scalar              minColorThre = Scalar(20, 20, 20);//Ĭ��LED����ɫ��Χ��ʹ��LEDģ���ʱ�������
	Scalar              maxColorThre = Scalar(250, 250, 250);
};
#endif
