This report presents some preliminary research work using LSTM neural networks to assess the impacts of air pollutants exposure on public health.

MIC-based Correlation Analysis
Mutual Information (MI)-based Maximal Information Coeffi cient (MIC) is used to evaluate the association between diff erent air pollutants and fi nd the most related factors for health consequence of interest, taking advantage of information entropy to capture both the linear and nonlinear relation.

 Daily mean values of temperature and air pollutants concentration and daily mortality for Chicago, IL, the U.S. from 1987 to 2000.
<img width="574" height="530" alt="2 2" src="https://github.com/user-attachments/assets/fdd78ca5-32fd-4222-ab5e-cde8ba0a040b" />
 Relations of diff erent air pollutants for Toronto, ON, Canada.
<img width="571" height="490" alt="2 3" src="https://github.com/user-attachments/assets/326ceef5-f501-43fd-a965-d03d0b7268ad" />
 MICs between air pollutants and daily mortality from diff erent causes for Chicago, IL, the U.S. 
<img width="532" height="724" alt="t2 1" src="https://github.com/user-attachments/assets/fe74bfaf-c95d-4373-ab3f-ee049ae81236" />

LSTM NetWork for Health Impact Assessment
An Long Short-Term Memory (LSTM) model is developed to assess the impacts of exposure to multiple air pollutants on health outcome of interest with weighted evaluation of distributed lags. An LSTM neural network is first designed to extract health outcome-related feature information from multi-pollutant exposure sequence with temporal dependence. Then estimation layers with weighted evaluation of the extracted features from the exposure that has distributed lags are constructed to assess the health consequence of interest.

