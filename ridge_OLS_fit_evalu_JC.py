import sys
import glob
import numpy as np
# import json
# import deepdish as dd
import pandas as pd
# import connectivity.io as cio
# from connectivity.data import Dataset
# import connectivity.constants as const
import matplotlib.pyplot as plt
import connectivity.model as model # import models
import connectivity.evaluation as ev # import evaluation methods
from numpy import linalg as LA #this package is used when calculating the norm
import math

#First, read the data
# read the cortex data as X
#X_cortex02 = pd.read_csv("X_cortex02_sess_weight.csv").iloc[:, 1:] 
X_cortex02 = pd.read_csv("X_cortex0162_sc1_02_sess_weight.csv").iloc[:, 1:] # read the large size data
#plt.imshow(X_cortex02)
#plt.show()
#XX_cortex=X_cortex02.fillna(0) # filling the missing values as 0

# read the cerebellum data as Y
#Y_cere02=pd.read_csv("Y_cere02_sess_weight.csv").iloc[:,1:]
Y_cere02=pd.read_csv("Y_cere_sc1_02_sess_weight_suit3.csv").iloc[:,1:] # read the large size data

#First, preprocessing the data by centering the data X and Y
X_mean=np.array([X_cortex02.mean(axis=0)]*int(X_cortex02.shape[0]))
X_preprocess=X_cortex02-X_mean

Y_mean=np.array([Y_cere02.mean(axis=0)]*int(Y_cere02.shape[0]))
Y_preprocess=Y_cere02-Y_mean


#Using ridge model to fit the data
ridge_model=model.L2regression(alpha=math.exp(4)) #using the default value alpha=e^8
#ridge_model=model.L2regression() #using the default value alpha=1
ridge_model.fit(X_preprocess, Y_preprocess)

#predicting on the same training dataset
Y_pred=ridge_model.predict(X_preprocess)
pred_err_training=LA.norm(Y_pred-Y_preprocess)
#239.81760570966875
#5.321512884163814

ridge_model.coef_ #print out the estimated coefficient matrix

#predicting on the different testset
X_cortex_sc2_02 = pd.read_csv("X_cortex0162_sc2_02_sess_weight.csv").iloc[:, 1:] #already fill the missing values as 0
#plt.imshow(X_cortex_sc2_02)
#plt.show()
Y_cere_sc2_02=pd.read_csv("Y_cere_sc2_02_sess_weight.csv").iloc[:,1:]


X_cortex_sc2_02_mean=np.array([X_cortex_sc2_02.mean(axis=0)]*int(X_cortex_sc2_02.shape[0]))
X_test=X_cortex_sc2_02-X_cortex_sc2_02_mean

Y_cere_sc2_02_mean=np.array([Y_cere_sc2_02.mean(axis=0)]*int(Y_cere_sc2_02.shape[0]))
Y_test=Y_cere_sc2_02-Y_cere_sc2_02_mean

X_test=X_cortex_sc2_02
Y_test=Y_cere_sc2_02

#test the prediction rate on the testset
Y_pred_test=ridge_model.predict(X_test)
pred_err_test=LA.norm(Y_pred_test-Y_test)
pred_err_test
#||Yo-Yp||_F:245.35603130225888; alpha=e^8
#245.66552392801853 for alpha=exp(4)
#238.56341456658043 for alpha=exp(6)
#296.692775862684; alpha=1
##For unweighted data:
#101.84245021759669


R, R_voxel=ev.calculate_R(Y_test, Y_pred_test) #R=0.40721647539359784; alpha=e^8
#R=0.44767823642130783 for alpha=exp(6)
#R=0.4417099218368717 for alpha=exp(4)
#0.34790520356665694; alpha=1
##For unweighted data
#0.30991223308346716
plt.plot(R_voxel)
plt.show()

R2, R2_voxel=ev.calculate_R2(Y_test, Y_pred_test)
plt.plot(R2_voxel)
plt.show()
# R2: 0.15115592704282554; alpha=e^8
# R2=0.19750538175109988 for alpha=exp(6)
#R2=0.14901310893720887 for alpha=exp(4)
#-0.24121892259830102; alpha=1
##for unweighted data
#0.09580864776569042

#############  OLS method  ####################

#Fitting the OLS model to estimate C
C_ols=LA.pinv(X_cortex02.T@X_cortex02)@X_cortex02.T@Y_cere02

#predict on the test data
Y_pred_ols=np.array(X_test)@np.array(C_ols)
pred_err_norm_ols=LA.norm(Y_pred_ols-Y_test)
pred_err_norm_ols
#||Yo-Yp||_F:320.28362778463594


R_ols, R_voxel_ols=ev.calculate_R(Y_test, Y_pred_ols)
plt.plot(R_voxel_ols)
plt.show()
#R_ols: 0.3229428699842788

R2_ols, R2_voxel_ols=ev.calculate_R2(Y_test, Y_pred_ols)
plt.plot(R2_voxel_ols)
plt.show()
# R2_ols: -0.44645165319109514

############ LRSmooth####################

#read the test set X and Y for sc2
X_sc2 = pd.read_csv("X_cortex0162_sc2_02_sess_weight.csv").iloc[:, 1:]
Y_sc2=pd.read_csv("Y_cere_sc2_02_sess_weight.csv").iloc[:,1:]

X_cortex_sc2_02_mean=np.array([X_cortex_sc2_02.mean(axis=0)]*int(X_cortex_sc2_02.shape[0]))
X_test=X_cortex_sc2_02-X_cortex_sc2_02_mean

Y_cere_sc2_02_mean=np.array([Y_cere_sc2_02.mean(axis=0)]*int(Y_cere_sc2_02.shape[0]))
Y_test=Y_cere_sc2_02-Y_cere_sc2_02_mean

#read the estimated C; Select R=5 as the optimal R
C_est=pd.read_csv("C_hat_R5.csv").iloc[:, 1:]
Y_pred_smooth=np.array(X_sc2)@np.array(C_est)

#Prediction error int terms of the norm of sum sqr err on test data
pred_err_smoth=LA.norm(Y_sc2-Y_pred_smooth)
#276.522532899882
#270.0833436831174 for C_R5(2)

R_LRS, R_voxel_LRS=ev.calculate_R(Y_sc2, Y_pred_smooth)
plt.plot(R_voxel_LRS)
plt.show()
#R_LRS: 0.29043048245516184
#0.3108376696532326 for C_R5(2)

R2_LRS, R2_voxel_LRS=ev.calculate_R2(Y_sc2, Y_pred_smooth)
plt.plot(R2_voxel_LRS)
plt.show()
#R2_LRS: -0.07819048959060271
#-0.028560986403830757 for C_R5(2)


#read the estimated C; Select R=1 as the optimal R
C_est=pd.read_csv("C_hat_R6_lambda10.csv").iloc[:, 1:]
Y_pred_smooth=np.array(X_sc2)@np.array(C_est)

#Prediction error int terms of the norm of sum sqr err on test data
pred_err_smoth=LA.norm(Y_sc2-Y_pred_smooth)
#For lambda=1
#262.4733050383467 for C_R1(2)
#264.96558055384264 for C_R2(2)
#262.7259079147047 for C_R3(2)
#267.0195786692181 for C_R4(2)
#270.0833436831174 for C_R5(2)
#271.07221653177356 for C_R6(2)
#273.2565304409173 for C_R7(2)
#276.93221889848496 for C_R8(2)
#278.21089202605935 for C_R9(2)
#281.2458784773353 for C_R10(2)
#283.9032126585382 for C_R11(2)
#285.66673987820576 for C_R12(2)
#290.96164118170947 for C_R13(2)
#295.85268400756144 for C_R14(2)
#311.95783711223066 for C_R15(2)
#314.84242859275497 for C_R16(2)
#317.46012968886066 for C_R17(2)
#320.9706512822698 for C_R18(2)
#321.9660543724963 for C_R19(2)
#325.4499722090814 for C_R20(2)

#For lambda=10
#R=1    264.312069
#R=2	265.506219
#R=3	263.332798
#R=4	266.353713
#R=5	269.738359
#R=6	274.830845
#R=7	276.659036
#R=8	278.533581
#R=9	282.970316
#R=10	286.225213
#R=11	289.570164
#R=12	291.608388
#R=13	293.512964
#R=14	299.167458
#R=15	302.450853
#R=16	323.833167
#R=17	326.589132
#R=18	329.617709
#R=19	330.870720
#R=20	333.368776

##For unweighted data
1
106.3058268632178
2
112.41272827248174
3
111.80305676987723
4
112.8985805086449
5
114.76579476598758
6
115.86975620647644
7
118.39646979998851
8
118.18026963739118
9
119.31829846009094
10
121.23552168883438
11
121.39159156846483
12
122.58011864178867
13
123.36687394685597
14
123.75209311841593
15
124.92678227582752


R_LRS, R_voxel_LRS=ev.calculate_R(Y_sc2, Y_pred_smooth)
plt.plot(R_voxel_LRS)
plt.show()
#For lambda=1
# 0.2720485090704462 for C_R1(2)
# 0.26982711212824856 for C_R2(2)
# 0.30983834526757764 for C_R3(2)
# 0.3033531174744737 for C_R4(2)
# 0.3108376696532326 for C_R5(2)
# 0.32219072196518744 for C_R6(2)
# 0.3241511127796172 for C_R7(2)
# 0.32181802915646696 for C_R8(2)
# 0.3303458210854234 for C_R9(2)
# 0.3262566019594292 for C_R10(2)
# 0.32844951737652406 for C_R11(2)
# 0.3317886301207888 for C_R12(2)
# 0.3212658271307138 for C_R13(2)
# 0.31762144892853694 for C_R14(2)
# 0.29003681612512905 for C_R15(2)
# 0.28747588508469124 for C_R16(2)
# 0.28563903515582917 for C_R17(2)
# 0.2815786215329702 for C_R18(2)
# 0.28653012457034455 for C_R19(2)
# 0.28482091121329994 for C_R20(2)

#For lambda=10
#R=1 0.20149752710184485           
#R=2 0.2285822410346615
#R=3 0.2707042656015946
#R=4 0.28270483020995824
#R=5 0.2791077801018707
#R=6 0.28566547670444226
#R=7 0.2989411627330832
#R=8 0.301406136387958
#R=9 0.3003815707258909
#R=10 0.30497046825541674
#R=11 0.30212751078500316
#R=12 0.3072165071891435
#R=13 0.30881780279884363
#R=14 0.29791274038157023
#R=15 0.2945257872387092
#R=16 0.26419757968615604
#R=17 0.2611033417343403
#R=18 0.25925807792556194
#R=19 0.2621206587158102
#R=20 0.2629499142899201

#lambda=0.1
1
0.2717892723705424
2
0.2716057478033129
3
0.2991479164812074
4
0.2961492934400395
5
0.29535804330681187
6
0.31068572593689026
7
0.31527739353704004
8
0.3108189551208772
9
0.31152560989455513
10
0.30663118905239556
11
0.3098511322449968
12
0.3175189765009777
13
0.293591767320412
14
0.28914966910383244
15
0.2749942741161381
16
0.2737559646531612
17
0.2710205587923313
18
0.26780193807443353
19
0.2714397710598153
20
0.2718693554801111

##For unweighted datat
1
0.15751102964771105
2
0.12416813332892
3
0.18118642587408096
4
0.18640811640263344
5
0.17911629585971714
6
0.17387803449858447
7
0.1798650511534968
8
0.20801592727300955
9
0.20565696350293428
10
0.22759786829898837
11
0.23483606817947034
12
0.22757878878860158
13
0.22678341951645103
14
0.23828202010351895
15
0.2467766779502684

R2_LRS, R2_voxel_LRS=ev.calculate_R2(Y_sc2, Y_pred_smooth)
plt.plot(R2_voxel_LRS)
plt.show()
#For lambda=1
# 0.028585182086782623 for C_R1(2)
# 0.010049753171095799 for C_R2(2)
# 0.026714513720414113 for C_R3(2)
# -0.005357811591534034 for C_R4(2)
# -0.028560986403830757 for C_R5(2)
# -0.03610664274576725 for C_R6(2)
# -0.05287192101338034 for C_R7(2)
# -0.08138767399941682 for C_R8(2)
# -0.09139686408635472 for C_R9(2)
# -0.11533871569921406 for C_R10(2)
# -0.13651470220418083 for C_R11(2)
# -0.15067797700656738 for C_R12(2)
# -0.19372948095344178 for C_R13(2)
# -0.23419979894705634 for C_R14(2)
# -0.3722279007023479 for C_R15(2)
# -0.3977224848581231 for C_R16(2)
# -0.42106133406092217 for C_R17(2)
# -0.45266372276919875 for C_R18(2)
# -0.4616877760202256 for C_R19(2)
# -0.49349207257898997 for C_R20(2)

#For lambda=10
#1 0.014926959879268176
#2 0.006005823464527937
#3 0.022212789378326936
#4 -0.00034994699262091977
#5 -0.025935045611425567
#6 -0.06503870754242302
#7 -0.07925523285462344
#8 -0.09393009258404028
#9 -0.12905787538352276
#10 -0.15518148187513936
#11 -0.18233914467967716
#12 -0.199042200445265
#13 -0.21475590697354696
#14 -0.2620110223256189
#15 -0.2898644462212592
#16 -0.4786898750305102
#17 -0.5039655980541411
#18 -0.5319885448500039
#19 -0.5436581021544895
#20 -0.5670551684189369

#lambda=0.1
1
0.03286565445878231
2
0.015002984685313026
3
0.015116956719127161
4
-0.007424284116905655
5
-0.03612470177101179
6
-0.046588732146014555
7
-0.06524301642564412
8
-0.10128956696596902
9
-0.13554703061161688
10
-0.161928681771355
11
-0.18868873891007287
12
-0.19971504397850048
13
-0.306345864378468
14
-0.36864436032432213
15
-0.4463959805102524
16
-0.4770296214766654
17
-0.5134543917650096
18
-0.564152445008655
19
-0.5713737909472061
20
-0.5946027094357833

#########plot lines to see the trend############
pred_err_norm=np.zeros(8)
R_line=np.zeros(8)
R2_line=np.zeros(8)

for i in range(8):
    r=i+1
    print(f"R={r}")
    C_est=pd.read_csv("C_hat_R"+str(r)+".csv").iloc[:, 1:]
    Y_pred_smooth=np.array(X_sc2)@np.array(C_est)
    pred_err_norm[i]=LA.norm(Y_sc2-Y_pred_smooth)
    R_line[i], R_voxel_LRS=ev.calculate_R(Y_sc2, Y_pred_smooth)
    R2_line[i], R2_voxel_LRS=ev.calculate_R2(Y_sc2, Y_pred_smooth)

pred_err_norm=pd.DataFrame(pred_err_norm)
pred_err_norm.to_csv("pred_err_norm.csv")
R_lines=pd.DataFrame(R_line)
R_lines.to_csv("R_lines.csv")
R2_lines=pd.DataFrame(R2_line)
R2_lines.to_csv("R2_lines.csv")

#plotting
x=np.array(range(8))+1
plt.plot(x, pred_err_norm)
plt.show()

plt.plot(x, R_line)
plt.show()

plt.plot(x, R2_line)
plt.show()




