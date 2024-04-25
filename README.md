MI-HMM-MAP
________________________________________
Official code implementation for MLA 21 paper Predicting density of serious crime incidents using a Multiple-Input Hidden Markov Maximization-a-Posteriori Model 

Robertson, Devon L., and Wayne S. Goodridge. "Predicting density of serious crime incidents using a Multiple-Input Hidden Markov Maximization a posteriori model." Machine Learning with Applications 7 (2022): 100231. https://doi.org/10.1016/j.mlwa.2021.100231

Environments
   
    Python: 3.6.3
    CPU: i7-1065G7
    RAM: 16GB

Requirements
 
    hmmlearn==0.3.2 \
    keras==3.3.2 \
    matplotlib==3.8.4 \
    numpy==1.26.4 \
    openpyxl==3.1.2 \
    pandas==2.2.2 \
    pipenv==2023.12.1 \
    pytest-warnings==0.3.1 \
    sklearn==0.0 \
    tensorflow==2.16.1

Google Colaboratory, is a powerful cloud-based platform that allows you to run Python code
without the need for any local installations. It’s an invaluable tool for data scientists,
machine learning engineers, and anyone who needs a free and easily accessible environment 
to develop and execute Python projects. In this blog post, we’ll guide you through the 
process of running Python projects on Google Colab.


Config: Setup Google Drive Colab Environment
1.	Prerequisites
	
       Google account
2.	Extract initial-setup.zip
3.	Navigate to Google Drive
4.	Upload initial-setup.ipynb
5.	Connect Colab To Google Drive
	Double left click the initial-setup.ipynb to open it
	You may need to do the following steps:
    -	At the top of the screen, click "Open with" and "Connect More Apps"
    -	Type colab into the search field and hit enter
    -	Click the "connect" button next to Google Colaboratory
6.	Create A Colab Notebooks Folder In Your Google Drive
    i.	Double left click the initial-setup.ipynb to open it
    ii.	At the top of the screen, click "Open with Google Colaboratory" 
    iii.	Under File menu Click Save a copy in Drive... 
    iv.	Close initial-setup.ipynb notebook 
    v.	Focus the copy of initial-setup.ipyb and move to step 6, below.
7.	Open initial-setup.ipynb Using Google Colaboratory
	If copy of initial-setup.ipynb is not open yet open with the following steps:
    i.	Navigate to your google drive 
    ii.	Navigate to the Colab Notebooks folder
    iii.	Double left click the Copy of initial-setup.ipynbto open it with "Open with Google Colaboratory" 
    iv.	Run Mount the drive cell ( click cell and press Shift+Enter )
    v.	Click the generated link 
    vi.	Choose your account
    vii.	Click Allow on the bottom
    viii.	Copy authorization code then press enter
    ix.	Run other cells ( click cell and press Shift+Enter )
Navigate your intro-to-python folder in your drive's Colab Notebooks folder

Other  Software To Possibly Consider For Offline Use
    Python 3 - The Anaconda Distribution.
    There are many ways to get Python on your computer, but the premier way of accomplishing this in the Data science world is by installing the Anaconda distribution.
    Which gives you access to Python 3, Jupyter Notebook, and a number of indispensable Data science libraries such as NumPy, Pandas, and MatplotLib

    Mac Os
        i.	Download the MacOs Installer for Python3
        ii.	Double-click the downloaded file and click continue to start the installation.
        iii.	Answer the prompts on the Introduction, Read Me and License screens.
        iv.	In Destination Select, click the Install button to install Anaconda in your Home User Directory.
        v.	(Optional) Microsoft VScode
        You will be given the option to install VScode. 
        vi.	Installation will be complete when you can search for and find the Anaconda Navigator and Anaconda Prompt  applications on your computer.

  Windows
        i.	Download the MacOs Installer for Python3
        ii.	 Double-click the downloaded file and click continue to start the installation.
        Do not launch the installer from within your Favorites Folder.
        iii.	 Read Licensing terms and click I Agree.  
        iv.	 Select an install for Just Me
        v.	Select a destination folder that is within your Home Directory.
        ex: c:\Users\your_name\Anaconda3
        vi.	DO add Anaconda to my Path Environment Variable
        vii.	Do Register Anaconda as my default Python 3.7
        viii.	(Optional) Microsoft VScode
        you will be given the option to install VScode. 
        ix.	After a successful installation you will see the “Thanks for installing Anaconda” dialog box.
        x.	You should now be able to find the Anaconda Navigator and Anaconda Prompt applications on your computer

Initialization For a given police district, the following Initialization should be considered for the execution of the hmm codes:

Arima:
                                                                                                            
startprob = np.array([0.5, 0.3, 0.2]) transmat= np.array([[0.5, 0.3, 0.2], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])

emissionprob = np.array([[0.5, 0.3, 0.2], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])

Barataria: 
startprob = np.array([0.2, 0.3, 0.5]) transmat= np.array([[0.4, 0.4, 0.2], [0.4, 0.4, 0.2], [0.4, 0.4, 0.2]])

emissionprob = np.array([[0.2, 0.3, 0.5], [0.3, 0.5, 0.2], [0.5, 0.3, 0.2]])

Besson Street: 
startprob = np.array([0, 0, 1]) transmat= np.array([[0, 0.1, 0.9], [0.1, 0, 0.9], [0.9, 0.1, 0]])

emissionprob = np.array([[0.2, 0.3, 0.5], [0.3, 0.5, 0.2], [0.5, 0.3, 0.2]])

Cunupia: 
startprob = np.array([0.5, 0.3, 0.2]) transmat= np.array([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])

emissionprob = np.array([[0.5, 0.3, 0.2], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])

Freeport: 
startprob = np.array([0.2, 0.3, 0.5]) transmat= np.array([[0.4, 0.4, 0.2], [0.4, 0.4, 0.2], [0.2, 0.4, 0.4]])

emissionprob = np.array([[0.2, 0.3, 0.5], [0.3, 0.5, 0.2], [0.5, 0.3, 0.2]])

Manzanilla: 
startprob = np.array([1, 0, 0]) transmat= np.array([[0.9, 0.1, 0], [0.1, 0.9, 0], [0, 0.1, 0.9]])

emissionprob = np.array([[0.5, 0.3, 0.2], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])

Morvant: 
startprob = np.array([0.2, 0.3, 0.5]) transmat= np.array([[0.6, 0.2, 0.2], [0.6, 0.2, 0.2], [0.6, 0.2, 0.2]])

emissionprob = np.array([[0.2, 0.3, 0.5], [0.3, 0.5, 0.2], [0.5, 0.3, 0.2]])

San Fernando: 
startprob = np.array([0.5, 0.3, 0.2]) transmat= np.array([[0.5, 0.3, 0.2], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])

emissionprob = np.array([[0.5, 0.3, 0.2], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])

Scarborough: 
startprob = np.array([0.2, 0.3, 0.5]) transmat= np.array([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])

emissionprob = np.array([[0.2, 0.3, 0.5], [0.3, 0.5, 0.2], [0.5, 0.3, 0.2]])

Siparia: 
startprob = np.array([0.2, 0.3, 0.5]) transmat= np.array([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])

emissionprob = np.array([[0.2, 0.3, 0.5], [0.3, 0.5, 0.2], [0.5, 0.3, 0.2]])

Portland Oregan USA:
startprob = np.array([0.5, 0.3, 0.2]) transmat= np.array([[0.1, 0.1, 0.8], [0.1, 0.8, 0.1], [0.8, 0.1, 0.1]])

emissionprob = np.array([[0.2, 0.2, 0.6], [0.2, 0.6, 0.2], [0.6, 0.2, 0.2]])

Train and Evaluate
        python3 hmm_MainPOS.py

        python3 mlp_MainPOS.py
        
        python3 rf_MainPOS.py
        
        python3 lstm_MainPOS.py
        
        python3 svm_MainPOS.py
        
        python3 dt_MainPOS.py
    
        python3 nb_MainPOS.py
    
        python3 lrm_MainPOS.py
        
        python3 hmm_MainPortland.py
    
        python3 mlp_MainPortland.py
        
        python3 rf_MainPortland.py
    
        python3 lstm_MainPortland.py
    
        python3 svm_MainPortland.py
    
        python3 dt_MainPortland.py
        
        python3 nb_MainPortland.py
    
        python3 lrm_MainPortland.py
