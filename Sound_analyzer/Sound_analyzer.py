#Measure pitch of all wav files in directory
import glob
import math
from pickle import TRUE
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
import os

# This is the function to measure voice pitch
def measurePitch(voiceID, f0min, f0max, unit, max_time, fmax):

    # call(sound, "To Harmonicity (gne)",minimum_frequency, maximum_frequency,bandwidth, step) -> change parameters based on male / female frequency ranges


    sound = parselmouth.Sound(voiceID) # read the sound
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    #print(pitch)
    
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    #print(meanF0)
    #stdevF0 = call(pitch, "Get standard deviation", 0 ,max_time, unit) # get standard deviation
    #print (stdevF0)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, max_time)

    # harmony noise ratio in cepstral HNR (CHNR) -> nerandu kaip paskaiciuoti naudojantis turimais praat objektais
    
    harmonicity = call(sound, "To Harmonicity (gne)", f0min, f0max, 1000, 80.)
    meanGNE = call(harmonicity, 'Get standard deviation...', 0, 0, 0, 0)

    num_points = call(pointProcess, "Get number of points")

    formant = call(sound, "To Formant (burg)", 0.04, 5, fmax, 0.02, 50.) # mazesnis langas turi buti, nei paduodamo garso iskarpos
    spectrogram = sound.to_spectrogram(window_length=0.04, maximum_frequency = 5500, time_step= 0.04, frequency_step = 50.0, window_shape = parselmouth.SpectralAnalysisWindowShape(1))

    formants = np.empty([5,num_points+1])
    formants_bandwidth = np.empty([5,num_points+1])
    formants_amplitude = np.empty([5,num_points+1])

    # IMPORTANT:
    #if meanF0 > 150:
    #        maxFormant = 5500  # women
    #    else:
    #        maxFormant = 5000  # men

    for point in range(1, num_points+1):
        t = call(pointProcess, "Get time from index", point)

        for form in range(1,5):
            f = formant.get_value_at_time(formant_number = form, time = t, unit = parselmouth.FormantUnit(0) ) #call(formant, "Get value at time", 1, t, unit, 'cubic') #SINC70 = <ValueInterpolation.SINC70: 3>
            f = f if not math.isnan(f) else 0
            formants[form-1:point-1] = f
            
            b = formant.get_bandwidth_at_time(formant_number = form, time = t, unit = parselmouth.FormantUnit(0) )
            formants_bandwidth[form-1:point-1] = b if not math.isnan(b) else 0

            a = spectrogram.get_power_at(t,f)        
            formants_amplitude[form-1:point-1] = a if not math.isnan(a) else 0

    # F1, F2 frequency
    formants = np.mean(formants, axis=1)

    # F1, F2 amplitude -> ar galima laikyti amplitude? cia is galios spektro istraukiau
    formants_amplitude = np.mean(formants_amplitude, axis=1)

    # Bandwidths 
    formants_bandwidth = np.mean(formants_bandwidth, axis=1)
    
    # F1 ir F2 amplitudziu skirtumas
    f_dif = formants_amplitude[0] / formants_amplitude[1]

    # MFCC 11 koef
    # http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
    mfcc = sound.to_mfcc(number_of_coefficients = 11, window_length = 0.02, time_step= 0.04, firstFilterFreqency = 100.0, distance_between_filters = 100.0, maximum_frequency = fmax)
    mfcc_f = mfcc.to_matrix_features(window_length = 0.01, include_energy = False) #mfcc.extract_features(window_length = 0.01, include_energy = False)
    #print(mfcc_f)
    frames = mfcc_f.get_number_of_rows()

    mfcc_matrix = np.zeros((frames, 11))

    for frame_no in range(1, frames+1):
        for coefficient_no in range(1, 11+1):
            coefficient_value = call(mfcc, 'Get value in frame', frame_no, coefficient_no)
            mfcc_matrix[frame_no-1, coefficient_no-1] = coefficient_value
    
    #print(mfcc_matrix)

    mfcc_matrix = mfcc_matrix[~np.isnan(mfcc_matrix).any(axis=1)] # drop NAN rows
    mfcc_matrix = np.mean(mfcc_matrix, axis=0) # PAKEISTI I ROOTS?? # cia jei keli frames, tai is ju vidurki isgauname
    #print(mfcc_matrix)

    #print(mfcc_matrix)

    # LPC ----------------------------------------------------------
    #You are advised not to use this command for formant analysis. 
    #For formant analysis, instead use Sound: To Formant (burg)..., which also works via LPC (linear predictive coding). 
    #This is because Sound: To Formant (burg)... lets you specify a maximum frequency, 
    #whereas the To LPC commands automatically use the Nyquist frequency as their maximum frequency. 
    #If you do use one of the To LPC commands for formant analysis, you may therefore want to downsample the sound first. 
    #For instance, if you want five formants below 5500 Hz but your Sound has a sampling frequency of 44100 Hz, 
    #you have to downsample the sound to 11000 Hz with the Sound: Resample... command. 
    #After that, you can use the To LPC commands, with a prediction order of 10 or 11.

    sound = sound.resample(new_frequency=11000,precision=50)

    lpc = call(sound, 'To LPC (burg)', 28, 0.02, 0.04, 50.) # 'autocorrelation', 'covariance', 'burg', 'maple'

    # LPC coef
    lpc_matrix = call(lpc, "Down to Matrix (lpc)")
    lpc_matrix = np.array(lpc_matrix.values).flatten()

    # LPC Formants
    # consult https://www.fon.hum.uva.nl/praat/manual/LPC__To_Formant.html
    tmp = call(lpc, "To Formant")
    lpc_formants = []
    # loop through formants
    t = call(pointProcess, "Get time from index", 1)
    t = t if not math.isnan(t) else 0.01 # centras iraso (window length = 0.02)
    #print(t)
    for num in range(1,14): 
        lpc_formants.append(tmp.get_value_at_time(formant_number = num, time = t, unit = parselmouth.FormantUnit(0) ) )
 
    #print(lpc_matrix)
    #print(lpc_formants.shape)

    # soft phonation index SPI

    #     3. Soft Phonation Index (SPI) 
    # SPI is the average ratio of the lower frequency harmonic 
    # energy in the range 70-1600Hz to the higher frequency 
    # harmonic energy in the range 1600-4200Hz. It is an 
    # indicator of how completely or tightly the vocal folds are 
    # adducted during phonation [7].

    # spectral tilt to calculate this? -> avg. energy in the low frequency components to the high frequencies

    spectrum = call(sound,"To Spectrum")

    SPI = call(spectrum,"Get band energy difference...",70,1600,1600,4200)
    #print(SPI)

    # CPPS
    power_cepstrogram = call(sound, "To PowerCepstrogram...", 75, 0.04, fmax, 50 )
    CPPS = call(power_cepstrogram, "Get CPPS...", "yes", 0.04, 0.0001, 75, 330, 0.05, "parabolic", 0.001, 0.05, "Exponential decay", "Robust slow")

    # harmony noise ratio in cepstral HNR (CHNR)
    # https://asa.scitation.org/doi/10.1121/1.421305
    # https://sci-hub.se/https://link.springer.com/chapter/10.1007/11613107_13
    # https://www.speech.kth.se/prod/publications/files/3270.pdf
    # PRAAT has RNR - Rharmonics to noise ratio

    power_cepstrum = call(power_cepstrogram, "To PowerCepstrum (slice)...", 0.2)
    RNR = call(power_cepstrum, "Get rhamonics to noise ratio...", 60, 333, 0.2)

    #print(hnr)
    #print(RNR)
    # Quefrency: represents time/ the period of vibration and is the horizontal axis
    # Rhamonics: the small peaks across the horizontal axis
    # Dominant rhamonic: has a greater amplitude than all of the others also known as he cepstral peak
    # Cepstral peak: the dominant rhamonic in the cepstrum The larger it is the more regular and stable the phonation is Can also apply to connected speech
    # Cepstral peak prominence: the amplitude of the cepstral peak
    # Smoothed cepstral peak prominence (CPPs)
    # the computer software calculated difference between the cepstral peak and the regression line at the cepstral peak

    #     4. Voice Turbulence Index (VTI)
    # VTI is a ratio of the spectral inharmonic high frequency 
    # within the range 1800-5800Hz to the spectral harmonic 
    # energy in the range 70-4200Hz, obtained from a single 
    # 1024 point block of the signal where the influence of the 
    # frequency and amplitude variation, voice breaks and sub 
    # harmonic components are minimal [6]. VTI measures 
    # relative energy of high frequency noise. It correlates with 
    # the turbulence caused by incomplete or loose adduction 
    # of the vocal folds [7]. 
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2639982/#:~:text=Voice%20Turbulence%20Index%20is%20an,sub%2Dharmonic%20components%20are%20minimal.
    # https://www.ijser.org/researchpaper/Normative-Voice-Data-On-Selected-Parameters-For-Young-Adults-Using-VISI-PITCH-IV.pdf
    # The western normative value for VTI given in the 
    # MDVP manual is 0.052 (SD=0.016) for males and 0.046 
    # (SD= 0.012) for females which is higher in comparison 
    # to the present study. In the MDVP norms, there is 
    # significant difference in the mean of VTI values between 
    # males and females which is not seen in the present study. 
    # https://orbi.umons.ac.be/bitstream/20.500.12907/29386/1/1-s2.0-S0892199719303285-main.pdf
    # https://www.ijopl.com/doi/IJOPL/pdf/10.5005/jp-journals-10023-1151

    #is return pasalintas stdevF0 nes per mazais gabaliukais sukarpyta nera is ko skaiciuoti
    return meanF0, hnr, localJitter, localShimmer, meanGNE, num_points, formants[0], formants[1], formants_amplitude[0], formants_amplitude[1], f_dif, mfcc_matrix, lpc_matrix, lpc_formants, SPI, CPPS, RNR

# create lists to put the results
file_list = []
window_no = []
mean_F0_list = []
#sd_F0_list = []
hnr_list = []
local_jitter_list = []
local_shimmer_list = []
gne_list = []
num_pounts_list = []
f1_list = []
f2_list = []
f1_a_list = []
f2_a_list = []
f_dif_list = []
mfcc_list = []
lpc_matrix_list = []
lpc_formants_list = []
SPI_list = []
CPPS_list = [] 
RNR_list = []

i = 0

rootdir = 'C:/bulk_insert'
# Go through all the wave files in the folder and measure features
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        wave_file = os.path.join(subdir, file) # 'C:/bulk_insert\Female\cancer\1957-a_h.wav'
        #print(os.path.join(subdir, file))

        if '.csv' not in wave_file:
            i = i +1
            print(i)
            sound = parselmouth.Sound(wave_file)
            sound =sound.resample(new_frequency=44100,precision=100)
            duration = 0.04 #sound.duration
            x = 0.00
            w_no = 0
            while x < duration:
                x_min = round(x,2) # start time
                x_max = round(x + 0.04,2) # end time
                w_no += 1 # window no

                if 'female' not in wave_file:
                    fmax = 5000
                else:
                    fmax = 5500

                # extract and window with Hanning
                w_sound = sound.extract_part(from_time=x_min, to_time=x_max, window_shape = parselmouth.WindowShape(4), preserve_times = False) # preserve_times=True) # hamming, 40ms, 20ms shift
                w_sound = parselmouth.Sound(w_sound)
                (meanF0, hnr, localJitter, localShimmer,meanGNE,num_points,f1,f2,f1_a,f2_a,f_dif,mfcc,lpc_matrix, lpc_formants, SPI, CPPS, RNR) = measurePitch(w_sound, 75, 600, "Hertz", 0.02,fmax)
                file_list.append(wave_file) # make an ID list
                window_no.append(w_no) # make an ID list
                mean_F0_list.append(meanF0) # make a mean F0 list
                #sd_F0_list.append(stdevF0) # make a sd F0 list
                hnr_list.append(hnr)
                local_jitter_list.append(localJitter)
                local_shimmer_list.append(localShimmer)
                #gne_list.append(meanGNE)
                num_pounts_list.append(num_points)
                f1_list.append(f1)
                f2_list.append(f2)
                f1_a_list.append(f1_a)
                f2_a_list.append(f2_a)
                f_dif_list.append(f_dif)
                mfcc_list.append(mfcc)
                lpc_matrix_list.append(lpc_matrix)
                lpc_formants_list.append(lpc_formants)
                SPI_list.append(SPI)
                CPPS_list.append(CPPS)
                RNR_list.append(RNR) # cia reikia paziureti kuris stulpelis isstumia lpc_f per daug
                x += 0.02

#print(len(file_list))
#print(len(lpc_matrix_list))
# pasalintas sd_F0_list
df = pd.DataFrame(np.column_stack([file_list,window_no, num_pounts_list, 
                                   f1_list, f1_a_list, f2_list, f2_a_list, 
                                   f_dif_list, mean_F0_list, hnr_list, local_jitter_list,
                                   local_shimmer_list, #gne_list,
                                   mfcc_list, lpc_matrix_list, lpc_formants_list, SPI_list, CPPS_list, RNR_list]) ,                           
columns=['voiceID', 'windowNo',
         'num_points', 'F1','F1_A', 
         'F2','F2_A', 'F1_f2_diff', 
         'meanF0Hz', 'HNR', 'lJitter',
         'lShimmer',#'GNE',
         'mfcc1','mfcc2','mfcc3','mfcc4','mfcc5','mfcc6','mfcc7','mfcc8','mfcc9','mfcc10','mfcc11',
         'lpc1','lpc2','lpc3','lpc4','lpc5','lpc6','lpc7','lpc8','lpc9','lpc10','lpc11','lpc12','lpc13','lpc14','lpc15','lpc16','lpc17','lpc18','lpc19','lpc20','lpc21','lpc22','lpc23','lpc24','lpc25','lpc26','lpc27','lpc28',
         'lpc_f1','lpc_f2','lpc_f3','lpc_f4','lpc_f5','lpc_f6','lpc_f7','lpc_f8','lpc_f9','lpc_f10','lpc_f11','lpc_f12','lpc_f13','lpc_f14',
         'SPI','CPPS','RNR'
         ])  #add these lists to pandas in the right order

# Write out the updated dataframe
df.to_csv("C:/bulk_insert/processed_results.csv", index=False)
# print(df)

# https://github.com/mit-teaching-systems-lab/dcss-confusion-analysis-agent/blob/main/feature_extraction_utils.py
# https://github.com/zgalaz/phd-thesis
# http://audiologistjobandnotes.blogspot.com/2012/01/mdvp-parameters-explaining-by.html
# https://wstyler.ucsd.edu/praat/UsingPraatforLinguisticResearchLatest.pdf
# https://www.phonetik.uni-muenchen.de/~hoole/kurse/akustikfort/anleitung_vq.pdf
# https://www.phon.ucl.ac.uk/courses/spsci/expphon/week3.php
# https://www.hilarispublisher.com/open-access/which-mathematical-and-physiological-formulas-are-describing-voicepathology-an-overview-2329-9126-1000253.pdf
# https://www.phonanium.com/wp-content/uploads/2020/11/2015-_-Journal-of-Voice-_-Objective-dysphonia-measures-in-the-program-Praat-CPPS-and-AVQI.pdf

# Kai Hamming (40 ms), 20 ms shift:
    # harmony noise ratio HNR +
    # harmony noise ratio in cepstral HNR (CHNR)
    # voice turbulence index VTI

#     4. Voice Turbulence Index (VTI)
# VTI is a ratio of the spectral inharmonic high frequency 
# within the range 1800-5800Hz to the spectral harmonic 
# energy in the range 70-4200Hz, obtained from a single 
# 1024 point block of the signal where the influence of the 
# frequency and amplitude variation, voice breaks and sub 
# harmonic components are minimal [6]. VTI measures 
# relative energy of high frequency noise. It correlates with 
# the turbulence caused by incomplete or loose adduction 
# of the vocal folds [7]. 

    # soft phonation index SPI +

#     3. Soft Phonation Index (SPI) 
# SPI is the average ratio of the lower frequency harmonic 
# energy in the range 70-1600Hz to the higher frequency 
# harmonic energy in the range 1600-4200Hz. It is an 
# indicator of how completely or tightly the vocal folds are 
# adducted during phonation [7].



    # normalized noise energy NNE

# Harmonicity is expressed in dB: if 99% of the
# energy of the signal is in the periodic part, and 1% is noise, the HNR is
# 10*log10(99/1) = 20 dB. 
    
    # glotal to noise exitation ratio GNE + 
    # shimmer +
    # jitter +
    # MFCC (11 verciu) +
    # LPC (28 verciu) + 
    # 14 formants from LPC +
    # F1 ir F2 daznis ir amplitude (kai spektras MGDF) + (spektro nustatymu neradau)
    # Santykis tarp F1 ir F2 amplitudziu +

# -------------------------------------------------------------------------------------------
# Kai Hamming (150 ms), 75 ms shift:
    # Relative average perturbation RAP
    # pitch perturbation quotient PPQ
    # amplitude perturbation quotient APQ

# Kai Hamming (55 ms), 27.5 ms shift:
    # correlation dimension D_2
    # Largest Lyapunov Exponent LLE
    # Lempel-Ziv Complexity LZC
    # Hurst Exponent
    # entropijos matavimai
    # Recurrence Period Density RPDE
    # Detrended Fluctuation Analysis DFA

# is Teagen Energy Operator TEO kai lyginami sveiki ir patologiniai balsai:
    # koreliacijos koef
    # euklido ir logaritminis atstumas
    # plotas po TEO konturu
