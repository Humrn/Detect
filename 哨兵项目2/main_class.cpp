#include<iostream>
#include "Detect.h"


using namespace std;
using namespace cv;


void main()
{
	Detect detect;
	detect.brightThre = 120;
	
	//////�쳵
	//detect.T_LED_COLOR = Red;
	//detect.isShowFloorFill = 1;
	
 //   //������
	//detect.br = 60;
	//detect.armorCenterColorThre = Scalar(150, 150, 150);
	//detect.T_ANGLE_THRE = 20;
	//detect.lowDifference = 5;
	//detect.upDifference = 5;
	//detect.videoPath = "�����زĺ쳵��ת-ev-+3.MOV";
 //   //��������
	//detect.videoPath = "�����زĺ쳵��ת-ev-0.MOV";
	////detect.videoPath = "��̨�زĺ쳵����-ev-0.MOV";
 //   //������
	//detect.videoPath = "�����زĺ쳵��ת-ev--3.MOV";


	////����
	detect.br = 30;
	detect.T_LED_COLOR = Blue;
	detect.isShowFloorFill = 1;
	detect.floorFillThre = 2  ;
	//��������
	detect.lowDifference = 4;
	detect.upDifference = 4;
	detect.erodeSize = 1;
	detect.armorCenterColorThre = Scalar(120, 120, 120);
	detect.videoPath = "�����ز�������ת-ev-0.MOV";
	//detect.videoPath = "BlueCar.avi  ";
	////������
	//detect.lowDifference = 5;
	//detect.upDifference = 5;
	//detect.armorCenterColorThre = Scalar(220, 220, 220);
	//detect.videoPath = "�����ز�������ת-ev-+3.MOV";
	////detect.videoPath = "�����ز�������Խ�-ev-+3.MOV";
	////������
	//detect.videoPath = "�����ز�������ת-ev--3.MOV";
	detect.getAttackTarget();
	//system("pause");
}


