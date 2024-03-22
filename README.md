MI-HMM-MAP
________________________________________
Official code implementation for MLA 21 paper Predicting density of serious crime incidents using a Multiple-Input Hidden Markov Maximization-a-Posteriori Model 

Robertson, Devon L., and Wayne S. Goodridge. "Predicting density of serious crime incidents using a Multiple-Input Hidden Markov Maximization a posteriori model." Machine Learning with Applications 7 (2022): 100231. https://doi.org/10.1016/j.mlwa.2021.100231

Environments
   
    Python: 3.6.3
   
    CPU: i7-1065G7
   
    RAM: 16GB

Requirements
 
    pandas==1.4.3
 
    matplotlib==2.2.3

    numpy==1.16.6

    scipy==1.4.1

    keras==2.2.5
 
    scikit-learn==1.1.1

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

Train and Evaluate
        python3 hmm_MainPOS.py

        python3 mlp_MainPOS.py
        
        python3 rf_MainPOS.py
        
        python3 rnn_MainPOS.py
        
        python3 svm_MainPOS.py
        
        python3 dt_MainPOS.py
    
        python3 nb_MainPOS.py
    
        python3 lrm_MainPOS.py
        
        python3 hmm_MainPortland.py
    
        python3 mlp_MainPortland.py
        
        python3 rf_MainPortland.py
    
        python3 rnn_MainPortland.py
    
        python3 svm_MainPortland.py
    
        python3 dt_MainPortland.py
        
        python3 nb_MainPortland.py
    
        python3 lrm_MainPortland.py
