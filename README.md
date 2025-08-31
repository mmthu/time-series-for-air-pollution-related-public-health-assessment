This report presents some preliminary research work using LSTM neural networks to assess the impacts of air pollutants exposure on public health.

1. MIC-based Correlation Analysis
Mutual Information (MI)-based Maximal Information Coeffi cient (MIC) is used to evaluate the association between diff erent air pollutants and fi nd the most related factors for health consequence of interest, taking advantage of information entropy to capture both the linear and nonlinear relation.

 Daily mean values of temperature and air pollutants concentration and daily mortality for Chicago, IL, the U.S.

<img width="574" height="530" alt="2 2" src="https://github.com/user-attachments/assets/fdd78ca5-32fd-4222-ab5e-cde8ba0a040b" />

 Relations of different air pollutants for Toronto, ON, Canada.
 
<img width="571" height="490" alt="2 3" src="https://github.com/user-attachments/assets/326ceef5-f501-43fd-a965-d03d0b7268ad" />

 MICs between air pollutants and daily mortality from diff erent causes for Chicago, IL, the U.S. 
 
<img width="532" height="724" alt="t2 1" src="https://github.com/user-attachments/assets/fe74bfaf-c95d-4373-ab3f-ee049ae81236" />

2. LSTM NetWork for Health Impact Assessment
An Long Short-Term Memory (LSTM) model is developed to assess the impacts of exposure to multiple air pollutants on health outcome of interest with weighted evaluation of distributed lags. An LSTM neural network is first designed to extract health outcome-related feature information from multi-pollutant exposure sequence with temporal dependence. Then estimation layers with weighted evaluation of the extracted features from the exposure that has distributed lags are constructed to assess the health consequence of interest.

LSTM network for feature extraction of sequential air pollutantion exposure information 

<img width="356" height="432" alt="3 2" src="https://github.com/user-attachments/assets/080ccb1d-b661-42a2-ac1c-28448afd9662" />

Health outcome assessment with weighted evaluation of exposure lags

<img width="382" height="168" alt="3 3" src="https://github.com/user-attachments/assets/50259072-9c5d-41b9-8281-c52ecad21645" />

Training losses for diff erent lengths of air pollution exposure sequence(m=1–12)

<img width="564" height="762" alt="3 6" src="https://github.com/user-attachments/assets/80448446-2d53-48b5-8a60-fcc7304eee10" />

Comparison between prediction and actual daily mortality (m=1–12)

<img width="563" height="797" alt="3 7" src="https://github.com/user-attachments/assets/07af7523-3238-4d08-89b4-319420594db1" />
