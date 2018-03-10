#include<iostream>
#include "Detect.h"


using namespace std;
using namespace cv;


void main()
{
	Detect detect;
	detect.brightThre = 120;
	
	//////红车
	//detect.T_LED_COLOR = Red;
	//detect.isShowFloorFill = 1;
	
 //   //高亮度
	//detect.br = 60;
	//detect.armorCenterColorThre = Scalar(150, 150, 150);
	//detect.T_ANGLE_THRE = 20;
	//detect.lowDifference = 5;
	//detect.upDifference = 5;
	//detect.videoPath = "步兵素材红车旋转-ev-+3.MOV";
 //   //正常亮度
	//detect.videoPath = "步兵素材红车旋转-ev-0.MOV";
	////detect.videoPath = "炮台素材红车侧面-ev-0.MOV";
 //   //低亮度
	//detect.videoPath = "步兵素材红车旋转-ev--3.MOV";


	////蓝车
	detect.br = 30;
	detect.T_LED_COLOR = Blue;
	detect.isShowFloorFill = 1;
	detect.floorFillThre = 2  ;
	//正常亮度
	detect.lowDifference = 4;
	detect.upDifference = 4;
	detect.erodeSize = 1;
	detect.armorCenterColorThre = Scalar(120, 120, 120);
	detect.videoPath = "步兵素材蓝车旋转-ev-0.MOV";
	//detect.videoPath = "BlueCar.avi  ";
	////高亮度
	//detect.lowDifference = 5;
	//detect.upDifference = 5;
	//detect.armorCenterColorThre = Scalar(220, 220, 220);
	//detect.videoPath = "步兵素材蓝车旋转-ev-+3.MOV";
	////detect.videoPath = "步兵素材蓝车左对角-ev-+3.MOV";
	////低亮度
	//detect.videoPath = "步兵素材蓝车旋转-ev--3.MOV";
	detect.getAttackTarget();
	//system("pause");
}


