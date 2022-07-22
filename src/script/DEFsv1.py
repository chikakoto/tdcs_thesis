#!/usr/bin/python
# coding: utf-8

# In[ ]:


'''
STEPS:
1.  DEFINE INPUT PARAMETER VARIABLES
2.  BRAIN EXTRACT T1 MAP AND DOWNSAMPLE-->VISUALLY INSPECT AND RERUN IF NECESSARY
3.  CONVERT DICOM TO NIFTI
4.  RENAME FILES, AND ORGANIZE INTO DIRECTORIES
5.  ALIGN B0 MAGNITUDE IMAGES TO T1-->VISUALLY INSPECT AND RERUN IF NECESSARY
6.  TRANSFORM T1 MASK TO NATIVE BN SPACES AND APPLY TO PHASE UNWRAPPING
7.  RESCALE & UNWRAP PHASE MAPS
8.  CREATE Bz FIELD MAPS
9.  FOR CONVENIENCE, CONVERT THE 12, 3D IMAGES FOR PHASE UNWRAP (rad and unscaled), PHASE WRAPPED, FIELD MAPS(rad/s and nT) into 4D IMAGES AND DELETE 3D FILES
10. METHOD 1: RUN LINEAR REGRESSION ON 12 ECHOES TO FIT FOR SLOPE (B FIELD), Y-INT, R, P, STANDARD ERROR
11. METHOD 2: CREATE FIELD DIFFERENCE MAPS, AVERAGE ACROSS ECHOES
12. QUALITY CONTROL

12. TRANSFROM Bc FIELDS TO COMMON T1 SPACE --> B[X,Y,Z], DO RESAMPLING, GAUSSIAN FILTERING, INTERPOLATION
13. DECOMPOSE BNsinT1 INTO BxyzsinT1
14. SUM BX'S BY'S BZ'S ACROSS THE 5 DIFFERENT ROTATIONS (normalize the sum???)
15. apply curl of B to get J(x,y,z)


8. CALCULATE SNR MAPS
9. CREATE WEIGHTED AVERAGE MAPS. WEIGHT BY SNR
10. UPSAMPLE AND BLUR (OPTIONAL)
11. OUTPUT VOXEL VALUES TO TEXT
12.PLOT THE EMPIRICAL DATA COMPARED TO THEORETICAL
'''


# In[1]:


import commands
import numpy as np
import nibabel as nib
import os
from sklearn import linear_model
from scipy import stats
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import subprocess
from skimage import data, img_as_float, color, exposure
from skimage.restoration import unwrap_phase
import loopfield as lf
import pandas as pd

# In[3]:


def letter_range(start, stop, step=1):
    """Yield a range of lowercase letters.""" 
    for ord_ in range(ord(start.lower()), ord(stop.lower()), step):
        yield chr(ord_)


# In[4]:


def t1(sub_path, t1_path, t1_dcm_name):
    cwd_path = sub_path+"/NIFTI/bmaps_numpy/"
    commands.getoutput("mkdir -p "+cwd_path+"/T1/")
    print commands.getoutput("dcm2niix -o "+cwd_path+"/T1/ -d N -p N "+t1_path+"/"+t1_dcm_name)
    commands.getoutput("mv "+cwd_path+"/T1/*.nii* "+cwd_path+"T1/T1.nii")
    commands.getoutput("3dresample -dxyz 4.0 4.0 4.0 -prefix "+cwd_path+"/T1/T1_4x3.nii.gz -input "+cwd_path+"/T1/T1.nii")
    #for human use bet below
    #commands.getoutput("bet "+cwd_path+"T1/T1_4x3.nii.gz "+cwd_path+"T1/T1_4x3_bet -Z -m")
    #for phantom, use commands below to make mask
    commands.getoutput("fslmaths "+cwd_path+"T1/T1_4x3.nii.gz -thrp 10 "+cwd_path+"T1/T1_4x3_bet.nii.gz")
    commands.getoutput("fslmaths "+cwd_path+"T1/T1_4x3_bet.nii.gz -div "+cwd_path+"T1/T1_4x3_bet.nii.gz "+cwd_path+"T1/T1_4x3_bet_mask.nii.gz")


# In[5]:


def dcm2niix(sub_path, base_runs, t1_path):
    cwd_path = sub_path+"/NIFTI/bmaps_numpy/"
    print "belfore dcm2nii"
    print cwd_path
    print sub_path+"/DICOM"
    #dcm2niix flags -d N -p N -f %p_%s
    commands.getoutput("dcm2niix -o "+cwd_path+" -f %s -d N -p N "+sub_path+"/DICOM/")
    #commands.getoutput("dcm2niix -o "+cwd_path+" -d N -p N "+sub_path+"/DICOM/")
    print "after dcm2nii"

    for base_run in base_runs:
        print "in loop"
        #declare var for base_run directory to shorten commands (used later by echo loops):
        bdir = cwd_path+"base_run_"+base_run+"/"
        phase = int(base_run) + 1
        phase = str(phase)
        commands.getoutput("mkdir "+cwd_path+"base_run_"+base_run)
        #rename and move mag1 and phase1 image if using e* convention
        #commands.getoutput("mv "+cwd_path+base_run+"_e1.nii "+bdir+"mag1.nii")
        #commands.getoutput("mv "+cwd_path+phase+"_e1.nii "+bdir+"phase1.nii")
        #rename and move phase1 image if using e1_ph convention:
        #commands.getoutput("mv "+cwd_path+phase+"_e1_ph.nii "+bdir+"phase1.nii")
        #rename and move mag1 and phase1 image if using abc convention
        #commands.getoutput("mv "+cwd_path+base_run+".nii "+bdir+"mag1.nii")
        #commands.getoutput("mv "+cwd_path+phase+".nii "+bdir+"phase1.nii")

        #rename and move mag2:12 and phase2:12 IMAGES if with suffix *e_2-12(6)
        #n12 = list(range(2,13))
        for j in range(1,13): 
            print j
            commands.getoutput("mv "+cwd_path+base_run+"_e"+str(j)+".nii "+bdir+"mag"+str(j)+".nii")
            commands.getoutput("mv "+cwd_path+phase+"_e"+str(j)+"_ph.nii "+bdir+"phase"+str(j)+".nii") 
            """#commands.getoutput("mv "+cwd_path+phase+"_e"+str(n12[j])+".nii "+bdir+"phase"+str(n12[j])+".nii")
        #sometimes dcm2niix doesn't use same suffix naming, in which case:
        #rename phase echo2-12 files that have suffix a-k to "phase_2-12":
        #abc = list(letter_range("a","l"))
        #n12 = list(range(2,13))
        #for j in range(0,11): 
            commands.getoutput("mv "+cwd_path+phase+abc[j]+".nii "+bdir+"phase"+str(n12[j])+".nii")
            commands.getoutput("mv "+cwd_path+base_run+abc[j]+".nii "+bdir+"mag"+str(n12[j])+".nii")"""


            


def cleanup(sub_path):
    cwd_path = sub_path+"/NIFTI/bmaps_numpy/"
    commands.getoutput("mkdir -p "+cwd_path+"/extras/")
    commands.getoutput("mv "+cwd_path+"/*.nii "+cwd_path+"/extras/")
    commands.getoutput("rm "+cwd_path+"/*.json")


# In[7]:


def bmap(sub_path, base_runs, echomax, tes, d):
    cwd_path = sub_path+"/NIFTI/bmaps_numpy/"
    if d==3:
        print commands.getoutput("3dresample -master "+cwd_path+"/base_run_"+str(base_runs[0])+"/mag1.nii -input "+cwd_path+"/T1/T1_4x3_bet_mask.nii.gz -prefix "+cwd_path+"/T1/T1_4x3_bet_mask2.nii.gz")
    if d==4:
        print commands.getoutput("3dresample -master "+cwd_path+"/base_run_"+str(base_runs[0])+"/mag1.nii -input "+cwd_path+"/T1/T1_4x3_bet_mask.nii.gz -prefix "+cwd_path+"/T1/T1_4x3_bet_mask3.nii.gz")


    echoes = range(1,echomax+1)
    echoess = range(1,echomax)
    print echoes
    for base_run in base_runs:
        bdir = cwd_path+"base_run_"+base_run+"/"
        for e in echoes:
            #STEP 7. RESCALE AND UNWRAP PHASE DATA
            #mask the original mag and phase data and convert to radians
            if d ==3:
                print commands.getoutput("fslmaths "+bdir+"phase"+str(e)+".nii -mul "+cwd_path+"/T1/T1_4x3_bet_mask2.nii.gz -mul 3.14159 -div 4096 "+bdir+"phase"+str(e)+"_rad -odt float")
            if d ==4:
                print commands.getoutput("fslmaths "+bdir+"phase"+str(e)+".nii -mul "+cwd_path+"/T1/T1_4x3_bet_mask3.nii.gz -mul 3.14159 -div 4096 "+bdir+"phase"+str(e)+"_rad -odt float")
    #try phase difference maps again (phase(te2)-phase(te1))
    for base_run in base_runs:
        bdir = cwd_path+"base_run_"+base_run+"/"
        for e in echoess:
            '''THIS IS ONE SUBROUTINE WHERE 1ST ECHO PHASE IS SUBTRACTED FROM EACH SUCCESSIVE ECHO. MIGHT BE
            CREATING MORE PHASE WRAP ERRORS. SHOULD MAKE OPTION ON WHAT KIND OF WAY TO CALCULATE FIELD MAPS:
            1. FROM EACH PHASEn/TEn, 
            2. SUBTRACTING (PHASEn-PHASE1)/(TEn-TE1) which is done here
            3. SUBRACTING PHASES FROM ONE CURRENT FROM ANOTHE ROR ZERO WHICH MAY BE GOOD FOR PHANTOM BUT BAD FOR HUMAN BECASUE OF MOTION
            commands.getoutput("fslmaths "+bdir+"phase"+str(e+1)+"_rad.nii.gz -sub "+bdir+"phase1_rad.nii.gz "+bdir+"phase"+str(e+1)+str(1)+"_rad.nii.gz")   
            img = nib.load(bdir+"phase"+str(e+1)+str(1)+"_rad.nii.gz")
            hdr = img.header
            data = img.get_data()
            s=np.shape(data)
            try:#try it as if the data is 4D
                data_unwr = np.full((s[0],s[1],s[2],s[3]),0.0)
                for i in range(0,data.shape[3]):
                    data_unwr[:,:,:,i] = unwrap_phase(data[:,:,:,i])
            except IndexError as error:#if it isn't 4d then we can assume it's 2d:
                data_unwr = np.full((s[0],s[1],s[2]),0.0)
                for i in range(0,data.shape[2]):
                    data_unwr[:,:,i] = unwrap_phase(data[:,:,i])
            img_unwr = nib.Nifti1Image(data_unwr,affine=None,header = hdr)
            nib.save(img_unwr, os.path.join(bdir+"/phase"+str(e+1)+str(1)+"_unwr_rad_np.nii.gz")) '''
            commands.getoutput("fslmaths "+bdir+"phase"+str(e+1)+"_rad.nii.gz -sub "+bdir+"phase"+str(e)+"_rad.nii.gz "+bdir+"phase"+str(e+1)+"_m_"+str(e)+"_rad.nii.gz")   
            img = nib.load(bdir+"phase"+str(e+1)+"_m_"+str(e)+"_rad.nii.gz")
            hdr = img.header
            data = img.get_data()
            s=np.shape(data)
            
            """THIS TRY AND EXCEPT WAS REPLACED ...NO MORE FOR LOOP...2020_02_29
            try:#try it as if the data is 4D
                data_unwr = np.full((s[0],s[1],s[2],s[3]),0.0)
                for i in range(0,data.shape[3]):
                    data_unwr[:,:,:,i] = unwrap_phase(data[:,:,:,i])
            except IndexError as error:#if it isn't 4d then we can assume it's 2d:
                data_unwr = np.full((s[0],s[1],s[2]),0.0)
                for i in range(0,data.shape[2]):
                    data_unwr[:,:,i] = unwrap_phase(data[:,:,i])"""
            if d == 3:
                data_unwr = unwrap_phase(data)#THIS LINE REPLACES TRY/EXCEPT ABOVE
            if d == 4:
                data_unwr = np.full((s[0],s[1],s[2],s[3]),0.0)
                for i in range(0,data.shape[3]):
                    data_unwr[:,:,:,i] = unwrap_phase(data[:,:,:,i])
            img_unwr = nib.Nifti1Image(data_unwr,affine=None,header = hdr)
            nib.save(img_unwr, os.path.join(bdir+"/phase"+str(e+1)+"_m_"+str(e)+"_unwr_rad_np.nii.gz"))
            #STEP 8. CREATE CENTER FREQUENCY OFFSET MAPS and CONVERT TO nT B FIELD OFFSET MAPS
            commands.getoutput("fslmaths "+bdir+"phase"+str(e+1)+"_m_"+str(e)+"_unwr_rad_np -div "+str((tes[e]-tes[e-1]))+" -mul 3731.34 "+bdir+"fmap"+str(e+1)+"_m_"+str(e)+"_nT")
            #shouldn't the constant be 3738.0077128...using gyro=267.522218744rad/sT x 10^6??

def unwrap4d(sub_path, base_runs, echomax):
    cwd_path = sub_path+"/NIFTI/bmaps_numpy/"
    for b in base_runs:
        bdir = cwd_path+"base_run_"+b+"/"
        a = ""
        for e in range(1,echomax):
            a += bdir+"/phase_"+str(e+1)+"_m_"+str(e)+"_rad.nii.gz"
        commands.getoutput("fslmerge -t "+bdir+"/phase4d_rad.nii.gz "+a)
        img = nib.load(bdir+"phase4d_rad.nii.gz")
        hdr = img.header
        data = img.get_data()
        s=np.shape(data)
        data_unwr = np.full((s[0],s[1],s[2],s[3]),0.0)
        for i in range(0,data.shape[3]):
            data_unwr[:,:,:,i] = unwrap_phase(data[:,:,:,i])
        img_unwr = nib.Nifti1Image(data_unwr,affine=None,header = hdr)
        nib.save(img_unwr, os.path.join(bdir+"/phase4d_unwr_rad_np.nii.gz"))
        #STEP 8. CREATE CENTER FREQUENCY OFFSET MAPS and CONVERT TO nT B FIELD OFFSET MAPS
        commands.getoutput("fslmaths "+bdir+"phase4d_unwr_rad_np -div "+str((tes[e]-tes[e-1]))+" -mul 3731.34 "+bdir+"fmap"+str(e+1)+"_m_"+str(e)+"_nT")
            #shouldn't the constant be 3738.0077128...using gyro=267.522218744rad/sT x 10^6??        
        
#note: you should tag the argument below at the end of the def above (bmap)
def collfmap(sub_path, base_runs, echomax):
    cwd_path = sub_path+"/NIFTI/bmaps_numpy/"
    for b in base_runs:
        bdir = cwd_path+"base_run_"+b+"/"
        s = ""
        a = ""
        for e in range(1,echomax):
            s += "-"+str(chr(e+96))+" "+bdir+"/fmap"+str(e+1)+"_m_"+str(e)+"_nT.nii.gz " 
            a += str(chr(e+96))+","
        print commands.getoutput("3dcalc "+s+"-expr 'median("+a[:-1]+")' -prefix "+bdir+"fmap_Median.nii.gz")
        commands.getoutput("3dcalc "+s+"-expr 'mean("+a[:-1]+")' -prefix "+bdir+"fmap_Mean.nii.gz")


# In[9]:


'''MAKE DIFFERENCE MAPS TO GET AT B FIELD ONLY FROM TDCS CURRENT'''
#note this is only for 3d not 4d epis sequence
def diffbmap(sub_path, bp, d):
    cwd_path = sub_path+"/NIFTI/bmaps_numpy/"
    for b in bp:
        print b
#        print bp[b][1]
        bdir = cwd_path+b[0]+'minus'+b[1]+"/"
        print bdir
        commands.getoutput("mkdir "+bdir)
        commands.getoutput("fslmaths "+cwd_path+"base_run_"+b[0]+"/fmap_Median.nii.gz -sub "+cwd_path+"base_run_"+b[1]+"/fmap_Median.nii.gz "+bdir+"/diff_fmap_Median.nii.gz")
        commands.getoutput("fslmaths "+cwd_path+"base_run_"+b[0]+"/fmap_Mean.nii.gz -sub "+cwd_path+"base_run_"+b[1]+"/fmap_Mean.nii.gz "+bdir+"/diff_fmap_Mean.nii.gz")
        if d == 4:
            commands.getoutput("fslmaths "+bdir+"diff_fmap_Median.nii.gz -Tmean "+bdir+"/diff_fmap_Median_Tmean.nii.gz")
            commands.getoutput("fslmaths "+bdir+"diff_fmap_Median.nii.gz -Tstd "+bdir+"/diff_fmap_Median_Tstd.nii.gz")
            commands.getoutput("fslmaths "+bdir+"diff_fmap_Median.nii.gz -Tmedian "+bdir+"/diff_fmap_Median_Tmedian.nii.gz") 
            
            
def theory_map(sub_path, position, normal, radius, current, bp):
    field = lf.Field(length_units = lf.cm,current_units = lf.A,field_units = lf.uT)
    # single-turn 12 cm z-oriented coil at origin
#position = [0.161, -1.63, 0.]#coordinate system here is -x,-y that of txt file output from field maps. correct by changing sign of center coordinates
#normal = [0., 0., 1.]
#radius = 6.
#current = -0.00200
    c = lf.Loop(position, normal, radius, current)
    # add loop to field
    field.addLoop(c);
    #MAKE THEORETICAL B FIELD MAPS
    #read in xyz ijk coordinates from i=0 masked image
    cwd_path = sub_path+"/NIFTI/bmaps_numpy/"
    for i in range(0,len(bp)):
        print i
        df_t=pd.DataFrame(); Bvals = list(); df_t_geo = pd.DataFrame(); df_Bvals = pd.DataFrame()
        bdir = cwd_path+bp[i][0]+'minus'+bp[i][1]+"/"
        print bdir
        #make mask
        commands.getoutput("fslmaths "+bdir+"diff_fmap_Mean.nii.gz -div "+bdir+"diff_fmap_Mean.nii.gz "+bdir+"mask.nii.gz")
        commands.getoutput("3dmaskdump -o "+bdir+"/mask.txt -xyz -nozero "+bdir+"mask.nii.gz")
        df_t = pd.read_csv(bdir+"/mask.txt", sep=" ", header=None)
        a = len(df_t)
        for j in range(0,a-1):
            bval = 1000*field.evaluate([float(df_t[3][j])/10.,float(df_t[4][j])/10.,float(df_t[5][j])/10.])
            Bvals.append(bval[2])

        if i ==0:
            print i
            df_t_geo = df_t
            df_t_geo = df_t_geo.drop(6,1)
            df_t_geo = df_t_geo.drop(5,1)
            df_t_geo = df_t_geo.drop(4,1)
            df_t_geo = df_t_geo.drop(3,1)
            Bvals_col = pd.Series(Bvals)
            df_Bvals = df_t_geo
            df_Bvals.insert(loc=3,column = 3,value=Bvals_col)
            df_Bvals.to_csv(bdir+"/Bvals.txt", header=None, index=None, sep=' ')
        #!!!!! here you use a different -master and -mask file for 3dundump--need to fix that to all be the same
            commands.getoutput("3dUndump -ijk -datum float -prefix "+bdir+"/Bvals.nii -master "+bdir+"/mask.nii.gz -mask "+bdir+"mask.nii.gz "+bdir+"/Bvals.txt")
        if i ==1:
            print i
            Bvals1 = [-1*Bvals[k] for k in range(0,len(Bvals))]
            Bvals1_col = pd.Series(Bvals1)
            df_t_geo = df_t
            df_t_geo = df_t_geo.drop(6,1)
            df_t_geo = df_t_geo.drop(5,1)
            df_t_geo = df_t_geo.drop(4,1)
            df_t_geo = df_t_geo.drop(3,1)
            df_Bvals1 = pd.DataFrame()
            df_Bvals1 = df_t_geo
            df_Bvals1.insert(loc=3,column = 3,value=Bvals1_col)
            df_Bvals1.to_csv(bdir+"/Bvals.txt", header=None, index=None, sep=' ')
        #!!!!! here you use a different -master and -mask file for 3dundump--need to fix that to all be the same
            commands.getoutput("3dUndump -ijk -datum float -prefix "+bdir+"/Bvals.nii -master "+bdir+"/mask.nii.gz -mask "+bdir+"mask.nii.gz "+bdir+"/Bvals.txt")
        if i ==2:
            print i
            Bvals2 = [2*Bvals[k] for k in range(0,len(Bvals))]
            Bvals2_col = pd.Series(Bvals2)
            df_t_geo = df_t
            df_t_geo = df_t_geo.drop(6,1)
            df_t_geo = df_t_geo.drop(5,1)
            df_t_geo = df_t_geo.drop(4,1)
            df_t_geo = df_t_geo.drop(3,1)
            df_Bvals2 = pd.DataFrame()
            df_Bvals2 = df_t_geo
            df_Bvals2.insert(loc=3,column = 3,value=Bvals2_col)
            df_Bvals2.to_csv(bdir+"/Bvals.txt", header=None, index=None, sep=' ')
        #!!!!! here you use a different -master and -mask file for 3dundump--need to fix that to all be the same
            commands.getoutput("3dUndump -ijk -datum float -prefix "+bdir+"/Bvals.nii -master "+bdir+"/mask.nii.gz -mask "+bdir+"mask.nii.gz "+bdir+"/Bvals.txt")


def theorVmeas(sub_path,xminusy,cur):
    cwd = sub_path+"/NIFTI/bmaps_numpy/"+xminusy+"/"
    commands.getoutput("3dcalc -a "+cwd+"/mask.nii.gz -prefix "+cwd+"/mask_erode1.nii.gz -b a+i -c a-i -d a+j -e a-j -f a+k -g a-k -expr 'a*(1-amongst(0,b,c,d,e,f,g))'")
    commands.getoutput("fslmaths "+cwd+"/mask_erode1.nii.gz -mul "+cwd+"/diff_fmap_Mean.nii.gz "+cwd+"/diff_fmap_Mean_erode1.nii.gz")
    commands.getoutput("fslmaths "+cwd+"/mask_erode1.nii.gz -mul "+cwd+"/Bvals.nii "+cwd+"/Bvals_erode1.nii.gz")
    img_meas = None
    img_theo = None
    img_meas = nib.load(cwd+"/diff_fmap_Mean_erode1.nii.gz")
    img_theo = nib.load(cwd+"/Bvals_erode1.nii.gz")
    img_meas = img_meas.slicer[:,:,15:48]
    img_theo = img_theo.slicer[:,:,15:48]
    #img_theo = nib.load(cwd+"/diff_fmap_Mean.nii.gz")
    data_meas = None
    data_theo = None
    data_meas = img_meas.get_data()
    data_theo = img_theo.get_data()
    x = None
    x = data_theo.flatten()
    print len(x)
    y = None
    y = data_meas.flatten()
    print len(y)
    m, b, r, p, st_er = stats.linregress(x,y) 
    print r
    plt.plot(x,y, 'o', markersize = 1)
    yfit = [b + m * xi for xi in x]
    yisx = [0 + 1 * xi for xi in x]
    plt.plot(x, yfit)
    plt.plot(x, yisx)
    if cur == 2:
        plt.axis([0, 150, 0, 150])
    if cur == -2:
        plt.axis([-150,0, -150, 0])
    if cur == 4:
        plt.axis([0,300,0,300])
    plt.xlabel("Theory (nT)", fontsize=16)
    plt.ylabel("MRI Measurement (nT)", fontsize=16)
    plt.text(0.5,0.5,"r = "+str(r)+"\nst_err = "+str(st_er)+" nT\ny = "+str(m)+"*x + "+str(b))
    plt.title('B field from 6cm Loop of Wire', fontsize = 20)
#plt.show()
#    %matplotlib inline
    plt.savefig(cwd+"Theo_v_Meas.svg", format="svg")
    plt.savefig(cwd+"Theo_v_Meas.png", format="png")
    plt.close()
    