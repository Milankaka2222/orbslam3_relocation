

#include "Initializer.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

#include "Optimizer.h"
#include "ORBmatcher.h"

#include<thread>
#include <include/CameraModels/Pinhole.h>

namespace ORB_SLAM3
{

/**
 * @brief 根据参考帧构造初始化器
 * 
 * @param[in] ReferenceFrame        参考帧
 * @param[in] sigma                 测量误差
 * @param[in] iterations            RANSAC迭代次数
 */
Initializer::Initializer(const Frame &ReferenceFrame, float sigma, int iterations)
{
    mpCamera = ReferenceFrame.mpCamera;
	
    mK = ReferenceFrame.mK.clone();

	
    mvKeys1 = ReferenceFrame.mvKeysUn;

    mSigma = sigma;
    mSigma2 = sigma*sigma;
	
    mMaxIterations = iterations;
}

/**
 * @brief 计算基础矩阵和单应性矩阵，选取最佳的来恢复出最开始两帧之间的相对姿态，并进行三角化得到初始地图点
 * Step 1 重新记录特征点对的匹配关系
 * Step 2 在所有匹配特征点对中随机选择8对匹配特征点为一组，用于估计H矩阵和F矩阵
 * Step 3 计算fundamental 矩阵 和homography 矩阵，为了加速分别开了线程计算 
 * Step 4 计算得分比例来判断选取哪个模型来求位姿R,t
 * 
 * @param[in] CurrentFrame          当前帧，也就是SLAM意义上的第二帧
 * @param[in] vMatches12            当前帧（2）和参考帧（1）图像中特征点的匹配关系
 *                                  vMatches12[i]解释：i表示帧1中关键点的索引值，vMatches12[i]的值为帧2的关键点索引值
 *                                  没有匹配关系的话，vMatches12[i]值为 -1
 * @param[in & out] R21                   相机从参考帧到当前帧的旋转
 * @param[in & out] t21                   相机从参考帧到当前帧的平移
 * @param[in & out] vP3D                  三角化测量之后的三维地图点
 * @param[in & out] vbTriangulated        标记三角化点是否有效，有效为true
 * @return true                     该帧可以成功初始化，返回true
 * @return false                    该帧不满足初始化条件，返回false
 */
bool Initializer::Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
                             vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated)
{
    

    
    mvKeys2 = CurrentFrame.mvKeysUn;

    
    mvMatches12.clear();
	
    mvMatches12.reserve(mvKeys2.size());
    
    
    mvbMatched1.resize(mvKeys1.size());
    
    
    for(size_t i=0, iend=vMatches12.size();i<iend; i++)
    {
		
        
        if(vMatches12[i]>=0)
        {
			
            
            mvMatches12.push_back(make_pair(i,vMatches12[i]));
			
            mvbMatched1[i]=true;
        }
        else
			
            mvbMatched1[i]=false;
    }

    
    const int N = mvMatches12.size();
    
    
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);

	
    vector<size_t> vAvailableIndices;
	
    for(int i=0; i<N; i++)
    {
        vAllIndices.push_back(i);
    }
    
    
    
    
    mvSets = vector< vector<size_t> >(mMaxIterations,vector<size_t>(8,0));

	
    DUtils::Random::SeedRandOnce(0);

	
    for(int it=0; it<mMaxIterations; it++)
    {
		
        vAvailableIndices = vAllIndices;

        
		
        for(size_t j=0; j<8; j++)
        {
            
            int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);
            
            int idx = vAvailableIndices[randi];
			
			
            mvSets[it][j] = idx;

            
            
            
            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }

    
    
 
    
    vector<bool> vbMatchesInliersH, vbMatchesInliersF;
	
    float SH, SF;
    
    cv::Mat H, F;

    
    
    thread threadH(&Initializer::FindHomography,	
				   this,							
				   ref(vbMatchesInliersH), 			
				   ref(SH), 						
				   ref(H));							
    
    thread threadF(&Initializer::FindFundamental,this,ref(vbMatchesInliersF), ref(SF), ref(F));

    

    
	
    threadH.join();
    threadF.join();

    
    
	
    float RH = SH/(SH+SF);

    

    float minParallax = 1.0; 

    cv::Mat K = static_cast<Pinhole*>(mpCamera)->toK();
    
    
    if(RH>0.40) 
	{
		
        return ReconstructH(vbMatchesInliersH,	
							H,					
							K,					
							R21,t21,			
							vP3D,				
							vbTriangulated,		
							minParallax,				
												
							50);				
    }
	else 
    {
		
        
        return ReconstructF(vbMatchesInliersF,F,K,R21,t21,vP3D,vbTriangulated,minParallax,50);
    }
	
    return false;
}

/**
 * @brief 计算单应矩阵，假设场景为平面情况下通过前两帧求取Homography矩阵，并得到该模型的评分
 * 原理参考Multiple view geometry in computer vision  P109 算法4.4
 * Step 1 将当前帧和参考帧中的特征点坐标进行归一化
 * Step 2 选择8个归一化之后的点对进行迭代
 * Step 3 八点法计算单应矩阵矩阵
 * Step 4 利用重投影误差为当次RANSAC的结果评分
 * Step 5 更新具有最优评分的单应矩阵计算结果,并且保存所对应的特征点对的内点标记
 * 
 * @param[in & out] vbMatchesInliers          标记是否是外点
 * @param[in & out] score                     计算单应矩阵的得分
 * @param[in & out] H21                       单应矩阵结果
 */
void Initializer::FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
{
    
	
    const int N = mvMatches12.size();

    
    
    
    
    
   

	
    vector<cv::Point2f> vPn1, vPn2;
	
    cv::Mat T1, T2;
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);
	
    cv::Mat T2inv = T2.inv();

    
	
    score = 0.0;
	
    vbMatchesInliers = vector<bool>(N,false);

    
	
    vector<cv::Point2f> vPn1i(8);
	
    vector<cv::Point2f> vPn2i(8);
	
    cv::Mat H21i, H12i;

    
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    
	
    for(int it=0; it<mMaxIterations; it++)
    {
        
		
        for(size_t j=0; j<8; j++)
        {
			
            int idx = mvSets[it][j];

            
			
            vPn1i[j] = vPn1[mvMatches12[idx].first];    
            vPn2i[j] = vPn2[mvMatches12[idx].second];   
        }

		
        
        
        
        
   
        cv::Mat Hn = ComputeH21(vPn1i,vPn2i);
        
        
        
        
        H21i = T2inv*Hn*T1;
		
        H12i = H21i.inv();

        
        currentScore = CheckHomography(H21i, H12i, 			
									   vbCurrentInliers, 	
									   mSigma);				

    
        
        if(currentScore>score)
        {
			
            H21 = H21i.clone();
			
            vbMatchesInliers = vbCurrentInliers;
			
            score = currentScore;
        }
    }
}

/**
 * @brief 计算基础矩阵，假设场景为非平面情况下通过前两帧求取Fundamental矩阵，得到该模型的评分
 * Step 1 将当前帧和参考帧中的特征点坐标进行归一化
 * Step 2 选择8个归一化之后的点对进行迭代
 * Step 3 八点法计算基础矩阵矩阵
 * Step 4 利用重投影误差为当次RANSAC的结果评分
 * Step 5 更新具有最优评分的基础矩阵计算结果,并且保存所对应的特征点对的内点标记
 * 
 * @param[in & out] vbMatchesInliers          标记是否是外点
 * @param[in & out] score                     计算基础矩阵得分
 * @param[in & out] F21                       从特征点1到2的基础矩阵
 */
void Initializer::FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
{
    

    
	
    const int N = vbMatchesInliers.size();

    
    
    
    
    

    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);
	
    cv::Mat T2t = T2.t();

    
	
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    
	
    vector<cv::Point2f> vPn1i(8);
    
    vector<cv::Point2f> vPn2i(8);
    
    cv::Mat F21i;
    
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    
    
    for(int it=0; it<mMaxIterations; it++)
    {
        
        
        for(int j=0; j<8; j++)
        {
            int idx = mvSets[it][j];

            
			
            vPn1i[j] = vPn1[mvMatches12[idx].first];        
            vPn2i[j] = vPn2[mvMatches12[idx].second];       
        }

        
        cv::Mat Fn = ComputeF21(vPn1i,vPn2i);

        
        
        
        
        F21i = T2t*Fn*T1;

        
        currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);

		
        if(currentScore>score)
        {
            
            F21 = F21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}


/**
 * @brief 用DLT方法求解单应矩阵H
 * 这里最少用4对点就能够求出来，不过这里为了统一还是使用了8对点求最小二乘解
 * 
 * @param[in] vP1               参考帧中归一化后的特征点
 * @param[in] vP2               当前帧中归一化后的特征点
 * @return cv::Mat              计算的单应矩阵H
 */
cv::Mat Initializer::ComputeH21(
    const vector<cv::Point2f> &vP1, 
    const vector<cv::Point2f> &vP2) 
{
    
    
    
    
    
    
    
    
    
    
    
    

	
    const int N = vP1.size();

    
    cv::Mat A(2*N,				
			  9,				
			  CV_32F);      	

	
    for(int i=0; i<N; i++)
    {
		
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

		
        A.at<float>(2*i,0) = 0.0;
        A.at<float>(2*i,1) = 0.0;
        A.at<float>(2*i,2) = 0.0;
        A.at<float>(2*i,3) = -u1;
        A.at<float>(2*i,4) = -v1;
        A.at<float>(2*i,5) = -1;
        A.at<float>(2*i,6) = v2*u1;
        A.at<float>(2*i,7) = v2*v1;
        A.at<float>(2*i,8) = v2;

		
        A.at<float>(2*i+1,0) = u1;
        A.at<float>(2*i+1,1) = v1;
        A.at<float>(2*i+1,2) = 1;
        A.at<float>(2*i+1,3) = 0.0;
        A.at<float>(2*i+1,4) = 0.0;
        A.at<float>(2*i+1,5) = 0.0;
        A.at<float>(2*i+1,6) = -u2*u1;
        A.at<float>(2*i+1,7) = -u2*v1;
        A.at<float>(2*i+1,8) = -u2;

    }

    
    cv::Mat u,w,vt;

	
    cv::SVDecomp(A,							
				 w,							
				 u,							
				 vt,						
				 cv::SVD::MODIFY_A | 		
				     cv::SVD::FULL_UV);		

	
    
    return vt.row(8).reshape(0, 			
							 3); 			
}


/**
 * @brief 根据特征点匹配求fundamental matrix（normalized 8点法）
 * 注意F矩阵有秩为2的约束，所以需要两次SVD分解
 * 
 * @param[in] vP1           参考帧中归一化后的特征点
 * @param[in] vP2           当前帧中归一化后的特征点
 * @return cv::Mat          最后计算得到的基础矩阵F
 */
cv::Mat Initializer::ComputeF21(
    const vector<cv::Point2f> &vP1, 
    const vector<cv::Point2f> &vP2) 
{
    
    
    
    

	
    const int N = vP1.size();

	
    cv::Mat A(N,9,CV_32F); 

    
    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(i,0) = u2*u1;
        A.at<float>(i,1) = u2*v1;
        A.at<float>(i,2) = u2;
        A.at<float>(i,3) = v2*u1;
        A.at<float>(i,4) = v2*v1;
        A.at<float>(i,5) = v2;
        A.at<float>(i,6) = u1;
        A.at<float>(i,7) = v1;
        A.at<float>(i,8) = 1;
    }

    
    cv::Mat u,w,vt;

    
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
	
    cv::Mat Fpre = vt.row(8).reshape(0, 3); 

    
    
    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

	
    w.at<float>(2)=0; 
    
    
    return  u*cv::Mat::diag(w)*vt;
}

/**
 * @brief 对给定的homography matrix打分,需要使用到卡方检验的知识
 * 
 * @param[in] H21                       从参考帧到当前帧的单应矩阵
 * @param[in] H12                       从当前帧到参考帧的单应矩阵
 * @param[in] vbMatchesInliers          匹配好的特征点对的Inliers标记
 * @param[in] sigma                     方差，默认为1
 * @return float                        返回得分
 */
float Initializer::CheckHomography(
    const cv::Mat &H21,                 
    const cv::Mat &H12,                 
    vector<bool> &vbMatchesInliers,     
    float sigma)                        
{   
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

	
    const int N = mvMatches12.size();

    const float h11 = H21.at<float>(0,0);
    const float h12 = H21.at<float>(0,1);
    const float h13 = H21.at<float>(0,2);
    const float h21 = H21.at<float>(1,0);
    const float h22 = H21.at<float>(1,1);
    const float h23 = H21.at<float>(1,2);
    const float h31 = H21.at<float>(2,0);
    const float h32 = H21.at<float>(2,1);
    const float h33 = H21.at<float>(2,2);

	
    const float h11inv = H12.at<float>(0,0);
    const float h12inv = H12.at<float>(0,1);
    const float h13inv = H12.at<float>(0,2);
    const float h21inv = H12.at<float>(1,0);
    const float h22inv = H12.at<float>(1,1);
    const float h23inv = H12.at<float>(1,2);
    const float h31inv = H12.at<float>(2,0);
    const float h32inv = H12.at<float>(2,1);
    const float h33inv = H12.at<float>(2,2);

	
    vbMatchesInliers.resize(N);

	
    float score = 0;

    
	
    const float th = 5.991;

    
    const float invSigmaSquare = 1.0/(sigma*sigma);

    
    
    
    for(int i=0; i<N; i++)
    {
		
        bool bIn = true;

		
        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        
        
        
        
        
        
		
        const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
        const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
        const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;

        
        const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);

        const float chiSquare1 = squareDist1*invSigmaSquare;

        
        if(chiSquare1>th)
            bIn = false;
        else
            
            score += th - chiSquare1;

        
        
        
        
        
        
		
        const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
        const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

        
        const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);
        const float chiSquare2 = squareDist2*invSigmaSquare;
 
        
        if(chiSquare2>th)
            bIn = false;
        else
            score += th - chiSquare2;   

        
        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

/**
 * @brief 对给定的Fundamental matrix打分
 * 
 * @param[in] F21                       当前帧和参考帧之间的基础矩阵
 * @param[in] vbMatchesInliers          匹配的特征点对属于inliers的标记
 * @param[in] sigma                     方差，默认为1
 * @return float                        返回得分
 */
float Initializer::CheckFundamental(
    const cv::Mat &F21,             
    vector<bool> &vbMatchesInliers, 
    float sigma)                    
{

    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

	
    const int N = mvMatches12.size();

	
    const float f11 = F21.at<float>(0,0);
    const float f12 = F21.at<float>(0,1);
    const float f13 = F21.at<float>(0,2);
    const float f21 = F21.at<float>(1,0);
    const float f22 = F21.at<float>(1,1);
    const float f23 = F21.at<float>(1,2);
    const float f31 = F21.at<float>(2,0);
    const float f32 = F21.at<float>(2,1);
    const float f33 = F21.at<float>(2,2);

	
    vbMatchesInliers.resize(N);

	
    float score = 0;

    
	
    
    const float th = 3.841;

    
    const float thScore = 5.991;

	
    const float invSigmaSquare = 1.0/(sigma*sigma);


    
    for(int i=0; i<N; i++)
    {
		
        bool bIn = true;

	    
        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

		
        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        
        
		const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;
    
        
        const float num2 = a2*u2+b2*v2+c2;

        const float squareDist1 = num2*num2/(a2*a2+b2*b2);
        
        const float chiSquare1 = squareDist1*invSigmaSquare;
		
        
        
        
        if(chiSquare1>th)
            bIn = false;
        else
            
            score += thScore - chiSquare1;

        
        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;

        
        const float num1 = a1*u1+b1*v1+c1;

        const float squareDist2 = num1*num1/(a1*a1+b1*b1);

        
        const float chiSquare2 = squareDist2*invSigmaSquare;

        
        if(chiSquare2>th)
            bIn = false;
        else
            score += thScore - chiSquare2;
        
        
        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }
    
    return score;
}

/**
 * @brief 从基础矩阵F中求解位姿R，t及三维点
 * F分解出E，E有四组解，选择计算的有效三维点（在摄像头前方、投影误差小于阈值、视差角大于阈值）最多的作为最优的解
 * @param[in] vbMatchesInliers          匹配好的特征点对的Inliers标记
 * @param[in] F21                       从参考帧到当前帧的基础矩阵
 * @param[in] K                         相机的内参数矩阵
 * @param[in & out] R21                 计算好的相机从参考帧到当前帧的旋转
 * @param[in & out] t21                 计算好的相机从参考帧到当前帧的平移
 * @param[in & out] vP3D                三角化测量之后的特征点的空间坐标
 * @param[in & out] vbTriangulated      特征点三角化成功的标志
 * @param[in] minParallax               认为三角化有效的最小视差角
 * @param[in] minTriangulated           最小三角化点数量
 * @return true                         成功初始化
 * @return false                        初始化失败
 */
bool Initializer::ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                            cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    
    
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;
    
    
    cv::Mat E21 = K.t()*F21*K;

    
    
    cv::Mat R1, R2, t;

    
    
    
    
    
    
    
    
    
    
    DecomposeE(E21,R1,R2,t);  

    cv::Mat t1=t;
    cv::Mat t2=-t;

    
    
    
    
	
    vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;

	
    vector<bool> vbTriangulated1,vbTriangulated2,vbTriangulated3, vbTriangulated4;

	
    float parallax1,parallax2, parallax3, parallax4;

	
    int nGood1 = CheckRT(R1,t1,							
						 mvKeys1,mvKeys2,				
						 mvMatches12, vbMatchesInliers,	
						 K, 							
						 vP3D1,							
						 4.0*mSigma2,					
						 vbTriangulated1,				
						 parallax1);					
    int nGood2 = CheckRT(R2,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D2, 4.0*mSigma2, vbTriangulated2, parallax2);
    int nGood3 = CheckRT(R1,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D3, 4.0*mSigma2, vbTriangulated3, parallax3);
    int nGood4 = CheckRT(R2,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D4, 4.0*mSigma2, vbTriangulated4, parallax4);

    
    int maxGood = max(nGood1,max(nGood2,max(nGood3,nGood4)));

	
    R21 = cv::Mat();
    t21 = cv::Mat();

    
    
    int nMinGood = max(static_cast<int>(0.9*N), minTriangulated);

	
    
    int nsimilar = 0;
    if(nGood1>0.7*maxGood)
        nsimilar++;
    if(nGood2>0.7*maxGood)
        nsimilar++;
    if(nGood3>0.7*maxGood)
        nsimilar++;
    if(nGood4>0.7*maxGood)
        nsimilar++;

    
    
    
    if(maxGood<nMinGood || nsimilar>1)	
    {
        return false;
    }


    
    
    

    
    if(maxGood==nGood1)
    {
		
        if(parallax1>minParallax)
        {
            
            vP3D = vP3D1;

			
            vbTriangulated = vbTriangulated1;

			
            R1.copyTo(R21);
            t1.copyTo(t21);
			
            
            return true;
        }
    }else if(maxGood==nGood2)
    {
        if(parallax2>minParallax)
        {
            vP3D = vP3D2;
            vbTriangulated = vbTriangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood3)
    {
        if(parallax3>minParallax)
        {
            vP3D = vP3D3;
            vbTriangulated = vbTriangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood4)
    {
        if(parallax4>minParallax)
        {
            vP3D = vP3D4;
            vbTriangulated = vbTriangulated4;

            R2.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }

    
    return false;
}

/**
 * @brief 用H矩阵恢复R, t和三维点
 * H矩阵分解常见有两种方法：Faugeras SVD-based decomposition 和 Zhang SVD-based decomposition
 * 代码使用了Faugeras SVD-based decomposition算法，参考文献
 * Motion and structure from motion in a piecewise planar environment. International Journal of Pattern Recognition and Artificial Intelligence, 1988 
 * 
 * @param[in] vbMatchesInliers          匹配点对的内点标记
 * @param[in] H21                       从参考帧到当前帧的单应矩阵
 * @param[in] K                         相机的内参数矩阵
 * @param[in & out] R21                 计算出来的相机旋转
 * @param[in & out] t21                 计算出来的相机平移
 * @param[in & out] vP3D                世界坐标系下，三角化测量特征点对之后得到的特征点的空间坐标
 * @param[in & out] vbTriangulated      特征点是否成功三角化的标记
 * @param[in] minParallax               对特征点的三角化测量中，认为其测量有效时需要满足的最小视差角（如果视差角过小则会引起非常大的观测误差）,单位是角度
 * @param[in] minTriangulated           为了进行运动恢复，所需要的最少的三角化测量成功的点个数
 * @return true                         单应矩阵成功计算出位姿和三维点
 * @return false                        初始化失败
 */
bool Initializer::ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{

    
    
    
    
    
    
    
    
    
    

    
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    
    
    

    
    
    
    
    
    
    
    

    cv::Mat invK = K.inv();
    cv::Mat A = invK*H21*K;

    
    
    
    
    
    
    
    cv::Mat U,w,Vt,V;
    cv::SVD::compute(A, w, U, Vt, cv::SVD::FULL_UV);

    
    V=Vt.t();

    
    
    float s = cv::determinant(U)*cv::determinant(Vt);
    
    
    float d1 = w.at<float>(0);
    float d2 = w.at<float>(1);
    float d3 = w.at<float>(2);

    
    if(d1/d2<1.00001 || d2/d3<1.00001) {
        return false;
    }


    
    
    vector<cv::Mat> vR, vt, vn;
    vR.reserve(8);
    vt.reserve(8);
    vn.reserve(8);

    
    
    
    
    
    
    
    
    
    

    
    
    
    

    float aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
    float aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
    float x1[] = {aux1,aux1,-aux1,-aux1};
    float x3[] = {aux3,-aux3,aux3,-aux3};


    
    
    
    
    
    
    
    
    float aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);

    float ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
    float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

    
    
    
    
    

    
    
    

    
    
    

    
    
    
    
    for(int i=0; i<4; i++)
    {
        
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=ctheta;
        Rp.at<float>(0,2)=-stheta[i];
        Rp.at<float>(2,0)=stheta[i];
        Rp.at<float>(2,2)=ctheta;

        
        cv::Mat R = s*U*Rp*Vt;

        
        vR.push_back(R);

        
        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=-x3[i];
        tp*=d1-d3;

        
        
        
        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        
        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        
        cv::Mat n = V*np;
        
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }
    
    
    float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);
    
    float cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
    
    float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

    
    for(int i=0; i<4; i++)
    {
        
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=cphi;
        Rp.at<float>(0,2)=sphi[i];
        Rp.at<float>(1,1)=-1;
        Rp.at<float>(2,0)=sphi[i];
        Rp.at<float>(2,2)=-cphi;

        
        cv::Mat R = s*U*Rp*Vt;
        
        vR.push_back(R);

        
        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=x3[i];
        tp*=d1+d3;

        
        cv::Mat t = U*tp;
        
        vt.push_back(t/cv::norm(t));

        
        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        
        cv::Mat n = V*np;
        
        if(n.at<float>(2)<0)
            n=-n;
        
        vn.push_back(n);
    }

    
    int bestGood = 0;
    
    int secondBestGood = 0;    
    
    int bestSolutionIdx = -1;
    
    float bestParallax = -1;
    
    vector<cv::Point3f> bestP3D;
    
    vector<bool> bestTriangulated;

    
    
    
    
    for(size_t i=0; i<8; i++)
    {
        
        float parallaxi;
        
        vector<cv::Point3f> vP3Di;
        
        vector<bool> vbTriangulatedi;
    
        
        int nGood = CheckRT(vR[i],vt[i],                    
                            mvKeys1,mvKeys2,                
                            mvMatches12,vbMatchesInliers,   
                            K,                              
                            vP3Di,                          
                            4.0*mSigma2,                    
                            vbTriangulatedi,                
                            parallaxi);                     
        
        
        
        if(nGood>bestGood)
        {
            
            secondBestGood = bestGood;
            
            bestGood = nGood;
            
            bestSolutionIdx = i;
            
            bestParallax = parallaxi;
            bestP3D = vP3Di;
            bestTriangulated = vbTriangulatedi;
        }
        
        else if(nGood>secondBestGood)
        {
            
            secondBestGood = nGood;
        }
    }



    
    
    
    
    
    if(secondBestGood<0.75*bestGood &&      
       bestParallax>=minParallax &&
       bestGood>minTriangulated && 
       bestGood>0.9*N)
    {
        
        vR[bestSolutionIdx].copyTo(R21);
        vt[bestSolutionIdx].copyTo(t21);
        
        vP3D = bestP3D;
        
        vbTriangulated = bestTriangulated;

        
        return true;
    }

    return false;
}

/** 给定投影矩阵P1,P2和图像上的匹配特征点点kp1,kp2，从而计算三维点坐标
 * @brief 
 * 
 * @param[in] kp1               特征点, in reference frame
 * @param[in] kp2               特征点, in current frame
 * @param[in] P1                投影矩阵P1
 * @param[in] P2                投影矩阵P2
 * @param[in & out] x3D         计算的三维点
 */
void Initializer::Triangulate(
    const cv::KeyPoint &kp1,    
    const cv::KeyPoint &kp2,    
    const cv::Mat &P1,          
    const cv::Mat &P2,          
    cv::Mat &x3D)               
{
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

	
    cv::Mat A(4,4,CV_32F);

	
    A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
    A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
    A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
    A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

	
    cv::Mat u,w,vt;
	
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
	
	
    x3D = vt.row(3).t();
	
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
}


/**
 * @brief 归一化特征点到同一尺度，作为后续normalize DLT的输入
 *  [x' y' 1]' = T * [x y 1]' 
 *  归一化后x', y'的均值为0，sum(abs(x_i'-0))=1，sum(abs((y_i'-0))=1
 *
 *  为什么要归一化？
 *  在相似变换之后(点在不同的坐标系下),他们的单应性矩阵是不相同的
 *  如果图像存在噪声,使得点的坐标发生了变化,那么它的单应性矩阵也会发生变化
 *  我们采取的方法是将点的坐标放到同一坐标系下,并将缩放尺度也进行统一 
 *  对同一幅图像的坐标进行相同的变换,不同图像进行不同变换
 *  缩放尺度是为了让噪声对于图像的影响在一个数量级上
 * 
 *  Step 1 计算特征点X,Y坐标的均值 
 *  Step 2 计算特征点X,Y坐标离均值的平均偏离程度
 *  Step 3 将x坐标和y坐标分别进行尺度归一化，使得x坐标和y坐标的一阶绝对矩分别为1 
 *  Step 4 计算归一化矩阵：其实就是前面做的操作用矩阵变换来表示而已
 * 
 * @param[in] vKeys                               待归一化的特征点
 * @param[in & out] vNormalizedPoints             特征点归一化后的坐标
 * @param[in & out] T                             归一化特征点的变换矩阵
 */
void Initializer::Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)                           
{
    

    
    float meanX = 0;
    float meanY = 0;

	
    const int N = vKeys.size();

	
    vNormalizedPoints.resize(N);

	
    for(int i=0; i<N; i++)
    {
		
        meanX += vKeys[i].pt.x;
        meanY += vKeys[i].pt.y;
    }

    
    meanX = meanX/N;
    meanY = meanY/N;

    
    float meanDevX = 0;
    float meanDevY = 0;

    
    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
        vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

		
        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }

    
    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;

    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;

    
    
    for(int i=0; i<N; i++)
    {
		
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }

    
    
    
    
    T = cv::Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
}

/**
 * @brief 用R，t来对特征匹配点三角化，并根据三角化结果判断R,t的合法性
 * 
 * @param[in] R                                     旋转矩阵R
 * @param[in] t                                     平移矩阵t
 * @param[in] vKeys1                                参考帧特征点  
 * @param[in] vKeys2                                当前帧特征点
 * @param[in] vMatches12                            两帧特征点的匹配关系
 * @param[in] vbMatchesInliers                      特征点对内点标记
 * @param[in] K                                     相机内参矩阵
 * @param[in & out] vP3D                            三角化测量之后的特征点的空间坐标
 * @param[in] th2                                   重投影误差的阈值
 * @param[in & out] vbGood                          标记成功三角化点？
 * @param[in & out] parallax                        计算出来的比较大的视差角（注意不是最大，具体看后面代码）
 * @return int 
 */
int Initializer::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                       const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers,
                       const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
{
    
	
    vbGood = vector<bool>(vKeys1.size(),false);
	
    vP3D.resize(vKeys1.size());

	
    vector<float> vCosParallax;
    vCosParallax.reserve(vKeys1.size());

    
    
    
    
 
    
    cv::Mat P1(3,4,				
			   CV_32F,			
			   cv::Scalar(0));	
	
    K.copyTo(P1.rowRange(0,3).colRange(0,3));
    
    cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);

    
    
    cv::Mat P2(3,4,CV_32F);
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
	
    P2 = K*P2;
    
    cv::Mat O2 = -R.t()*t;

	
    int nGood=0;

	
    for(size_t i=0, iend=vMatches12.size();i<iend;i++)
    {

		
        if(!vbMatchesInliers[i])
            continue;

        
        
        const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
        const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
		
        cv::Mat p3dC1;

        
        Triangulate(kp1,kp2,	
					P1,P2,		
					p3dC1);		

		
        
        if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
        {
			
            vbGood[vMatches12[i].first]=false;
			
            continue;
        }

        
        

        
        cv::Mat normal1 = p3dC1 - O1;
		
        float dist1 = cv::norm(normal1);

		
        cv::Mat normal2 = p3dC1 - O2;
		
        float dist2 = cv::norm(normal2);

		
        float cosParallax = normal1.dot(normal2)/(dist1*dist2);

        
        
        
        
        
        if(p3dC1.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        
        
        cv::Mat p3dC2 = R*p3dC1+t;	
		
        if(p3dC2.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        
        
        cv::Point2f uv1 = mpCamera->project(p3dC1);
		
        float squareError1 = (uv1.x-kp1.pt.x)*(uv1.x-kp1.pt.x)+(uv1.y-kp1.pt.y)*(uv1.y-kp1.pt.y);

        
        if(squareError1>th2)
            continue;

        
        cv::Point2f uv2 = mpCamera->project(p3dC2);
        float squareError2 = (uv2.x-kp2.pt.x)*(uv2.x-kp2.pt.x)+(uv2.y-kp2.pt.y)*(uv2.y-kp2.pt.y);

        
        if(squareError2>th2)
            continue;

        
        
        vCosParallax.push_back(cosParallax);
		
        vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2));
		
        nGood++;

		
		
        if(cosParallax<0.99998)
            vbGood[vMatches12[i].first]=true;
    }
    
    if(nGood>0)
    {
        
        sort(vCosParallax.begin(),vCosParallax.end());

        
		
		
        size_t idx = min(50,int(vCosParallax.size()-1));
		
        parallax = acos(vCosParallax[idx])*180/CV_PI;
    }
    else
		
        parallax=0;

	
    return nGood;
}

/**
 * @brief 分解Essential矩阵得到R,t
 * 分解E矩阵将得到4组解，这4组解分别为[R1,t],[R1,-t],[R2,t],[R2,-t]
 * 参考：Multiple View Geometry in Computer Vision - Result 9.19 p259
 * @param[in] E                 本质矩阵
 * @param[in & out] R1          旋转矩阵1
 * @param[in & out] R2          旋转矩阵2
 * @param[in & out] t           平移向量，另外一个取相反数
 */
void Initializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
{

    
	
    cv::Mat u,w,vt;
	
    cv::SVD::compute(E,w,u,vt);

    u.col(2).copyTo(t);
    t=t/cv::norm(t);

    
    
    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1)=-1;
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;

	
    R1 = u*W*vt;
	
    if(cv::determinant(R1)<0)
        R1=-R1;

	
    R2 = u*W.t()*vt;
	
    if(cv::determinant(R2)<0)
        R2=-R2;
}

} 
