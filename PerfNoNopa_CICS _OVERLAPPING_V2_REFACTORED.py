import pandas as pd
import datetime
import time
import numpy as np
import tkinter as tk

from numpy import NAN, isnan
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine
from tkinter import *
from tkinter import messagebox
from tkinter.ttk import Progressbar

#Styles 
LARGEFONT =("Verdana", 35)
BACKGROUND = '#A9A9A9'
BUTTON = '#33CC00'
ACTIVEBUTTON = '#339900'
BUTTONFRAME = '#D3D3D3'
GRAPHFRAME = '#A9A9A9'
USERFOLDER = ''
FILENAME = '\CLEANED_DATA.csv' #OUTPUT FILE NAME
MAX_DIFF = 0.9
MIN_DIFF = 0.1

"""
AUTHOR:@Nicholas Ackerman
DATE: 04/01/2021
PURPOSE: This is a class function to control the connection to snowflake. The user currently just needs to send in their username. However in 
future updates the user might be requires to send in the DW or Production warehouse they wish to access. Currently the class opens snowflake in 
and external browser, 
REFERENCES: init, connectionOpen, ConnectionClose, ConnectionTest, QuerytoSF
"""
class SFConnection:
    def __init__(self):
        self.name = ''
        self.connection = ''

    def open_conn(self):
        username = self.name

        engine = create_engine(URL(
        account = '',
        user = username,
        password = '',
        database = 'DW_PRD',
        schema = '',
        warehouse = '',
        role='',
        authenticator='externalbrowser'
        ))
        self.connection = engine.connect()
        self.connection.begin()
        print("Connection Opened")

    def test_conn(self):
        try:
            self.open_conn()
            results = self.connection.execute('select current_version()').fetchone()
            print("Snowflake connected to Version: {}".format(results[0]))
            return("CONNECTED")
        except Exception as e:
            print("Snowflake connection Error")
            return("ERR")

    def run_query(self, sql):
        self.connection.execute(sql)

    def query_to_df(self, sql):
        #results = self.connection.execute(sql)
        return pd.read_sql_query(sql, self.connection)
        
    def close_conn(self):
        self.connection.close()
        print("Connection Closed")
    def add_name(self, name):
        self.name = name
        
class MyApp(tk.Tk):
     
    # __init__ function for class MyApp
    def __init__(self, *args, **kwargs):
         
        # __init__ function for class Tk
        tk.Tk.__init__(self, *args, **kwargs)

        #shared variables across the classes in a dictioanry form
        self.shared_data = {
            "CONN": SFConnection()
        }
        # creating a container
        container = tk.Frame(self, height= 25, width= 25)
        container.pack(side = "top", fill = "both", expand = True)
  
        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0,weight = 1)

        # initializing frames to an empty array
        self.frames = {} 
        # iterating through a tuple consisting of the different page layouts
        for F in (LoginPage, StartPage):
  
            frame = F(container, self)
  
            # initializing frame of that object from startpage,
            self.frames[F] = frame
  
            frame.grid(row = 0, column = 0, sticky ="nsew")
  
        self.show_frame(LoginPage)
  
    # to display the current frame passed as parameter
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

# first window frame startpage
class LoginPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, background = BACKGROUND)

        self.controller = controller

        # label of frame Layout 2
        label = tk.Label(self, text ="LOGIN", font = LARGEFONT, background= BACKGROUND)
        self.btn_1 = tk.Button(self, text="LOGIN", background=BUTTON, activebackground=ACTIVEBUTTON, command = self.VerifyLogin)
        self.txt_input = tk.Entry(self, textvariable="username")

        self.controller.bind('<Return>', self.callback)

        # putting the button in its place by using grid
        label.grid(row = 0, column = 0, padx = 10, pady = 10)
        self.txt_input.grid(row = 2, column = 0, padx = 10, pady = 10)
        self.btn_1.grid(row = 3, column = 0, padx = 10, pady = 10)

    def callback(self,event):
        #if self.txt_input.get() != '':
        #TODO this is not robust
        self.VerifyLogin()

    def VerifyLogin(self):
        connection = self.controller.shared_data["CONN"]
        connection.add_name(self.txt_input.get().upper())

        if connection.test_conn() == "CONNECTED":
            messagebox.showinfo("Information","SUCCESS!!")
            self.controller.show_frame(StartPage)
        else:
            messagebox.showerror("Warning","INVALID USERNAME: PLEASE TRY AGAIN")
            print("USERNAME ERROR")

# first window frame startpage
class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, background = BACKGROUND)

        self.conn = controller.shared_data["CONN"]

        # label of frame Layout 2
        label = tk.Label(self, text ="PERF NO NOPA", font = LARGEFONT, background= BACKGROUND)

        self.fr_data = tk.Frame(self, relief=tk.RAISED, background=BUTTONFRAME, bd=2)
        self.txt_input_sd = tk.Entry(self.fr_data,width=25, textvariable="START DATE")
        self.txt_input_ed = tk.Entry(self.fr_data,width=25, textvariable="END DATE")
        self.label_sd = tk.Label(self.fr_data, text ="START DATE [YYYY-MM-DD]",width=25, font = LARGEFONT, background= BUTTONFRAME)
        self.label_ed = tk.Label(self.fr_data, text ="END DATE [YYYY-MM-DD]",width=25, font = LARGEFONT, background= BUTTONFRAME)
        self.btn_1 = tk.Button(self.fr_data, text="RUN PROGRAM", background=BUTTON, activebackground=ACTIVEBUTTON, font = LARGEFONT, command = self.run_process)

        #progress bar
        self.fr_progress = tk.Frame(self, relief=tk.RAISED, background=BUTTONFRAME, bd=2)
        self.pbar = Progressbar(self.fr_progress, length=400, s='black.Horizontal.TProgressbar')
        self.label_prog = tk.Label(self.fr_progress, text ="PROGRESS", font = LARGEFONT, background= BUTTONFRAME)
        self.pbar['value'] = 0
        self.pbar.grid(row = 4, column = 1, padx = 10, pady = 10)
        self.label_prog.grid(row=4, column = 0, padx = 5, pady = 5)

        # putting the button in its place by using grid
        label.grid(row = 0, column = 0, padx = 10, pady = 10)
        
        self.fr_data.grid(row=1,column=0, sticky="ew")
        self.fr_progress.grid(row=2,column=0, sticky="ew")
        self.txt_input_sd.grid(row=1, column = 1, padx = 5, pady = 5)
        self.txt_input_ed.grid(row=2, column = 1, padx = 5, pady = 5)
        self.label_sd.grid(row=1, column = 0, padx = 5, pady = 5)
        self.label_ed.grid(row=2, column = 0, padx = 5, pady = 5)
        self.btn_1.grid(row = 3, column = 0, padx = 10, pady = 10)

    def add_progressbar(self, value):
        self.pbar['value'] = self.pbar['value'] + value
        self.update()

    def run_process(self):
        #STEP 1: produce queries for tool
        sqlList = CreateSnowflakeQueries(str(self.txt_input_sd.get()), str(self.txt_input_ed.get()))
        print("Connection Completed - Queries being Pulled")

        #STEP 2: Run queries and save data as CSV,
        self.export_sql_to_query(sqlList)

        #STEP 3: Open CSV FILES/Analyze
        print("Analysis Function Running")
        self.open_csv()
        messagebox.showinfo("Information","SUCCESS!!")
        self.pbar['value'] = 0
        
    def export_sql_to_query(self, sqlList):
        #namesList = ["delete", "insert", "0_1", "MKDN_PRICE", "MKDN_QTY_0_1", "RETAIL_PRICE", "MKDN_QTY"]
        namesList = ["0_1", "MKDN_PRICE", "MKDN_QTY_0_1", "RETAIL_PRICE", "MKDN_QTY"]
        count = 0
        for query in sqlList:
            #TODO ERROR
            #if count <= 1:
            #    self.conn.run_query(query)
            #else:
            self.conn.query_to_df(query).to_csv(USERFOLDER + '/PerfNoNopa_GUI/CSV_FILES/' + namesList[count] + '.csv')
            print("Query " + str(count) +" Complete " + "- " + namesList[count])
            self.add_progressbar(100/len(sqlList))
            count+=1

        self.conn.close_conn()

    def format_dataframes(self, csv_file, RemoveCICS):
        #REFORMATTING ROWS
        data = pd.read_csv (csv_file)
        dfOFFER_TEST = pd.DataFrame(data.iloc[:, 2])

        #removing duplicate CIC values
        df = pd.DataFrame(data)
        dfTEMP = df[~df.cic.isin(dfOFFER_TEST.cic)]
        RemoveCICS = dfTEMP['cic'].values.tolist()

        #indexing the dataframe 
        df = df.set_index('cic')
        df.drop(RemoveCICS, inplace = True)
        df = df.reset_index()
        df = df.iloc[:, 4:]
        df = df.fillna(0)

        return df

    def open_csv(self):

        data = pd.read_csv (USERFOLDER +'/PerfNoNopa_GUI/CSV_FILES/0_1.csv')
        dfOFFER = pd.DataFrame(data.iloc[:, 5:])
        dfOFFER = dfOFFER.fillna(0)

        #4 variables at the front of the dataframes
        dfIndex = pd.DataFrame(data.iloc[:, 1:5])

        dfMKDN = self.format_dataframes(USERFOLDER +'/PerfNoNopa_GUI/CSV_FILES/MKDN_PRICE.csv')
        dfRETAIL = self.format_dataframes(USERFOLDER +'/PerfNoNopa_GUI/CSV_FILES/RETAIL_PRICE.csv')
        dfQTY = self.format_dataframes(USERFOLDER +'/PerfNoNopa_GUI/CSV_FILES/MKDN_QTY_0_1.csv')
        dfIQTY = self.format_dataframes(USERFOLDER +'/PerfNoNopa_GUI/CSV_FILES/MKDN_QTY.csv')
        
        #sum the rows of the dataframe so that they can be added to the dataframe
        dfIQTY = dfIQTY.mul(dfQTY)
        dfIQTY = dfIQTY.mul(dfOFFER)
        dfIQTY  = dfIQTY.fillna(0)

        #Eliminate offers from dataframes
        dfMKDN = dfMKDN.mul(dfOFFER)        
        dfRETAIL = dfRETAIL.mul(dfOFFER)

        #eliminate areas where mkdn qty not greater than 10
        dfMKDN = dfMKDN.mul(dfQTY)        
        dfRETAIL = dfRETAIL.mul(dfQTY)

        #created price difference
        dfPriceDiff = dfMKDN.div(dfRETAIL)

        #round to two decimal poitns
        dfPriceDiff = dfPriceDiff.round(2)

        #reindex the dataframe
        dfPriceDiff['unit'] = dfIQTY.sum(axis=1)
        dfPriceDiff['info'] = dfIndex['cic'].map(str)
        dfPriceDiff['div'] = dfIndex['div'].map(str)
        dfPriceDiff['cic'] = dfIndex['cic'].map(str)
        dfPriceDiff['offer'] = dfIndex['offer'].map(str)
        dfPriceDiff = dfPriceDiff.set_index('info')

        #complete analysis of dataframe
        self.analyze_dataframe(dfPriceDiff)

    def analyze_dataframe(self, dfPriceDiff):
        #remove rows above a certain average--- This might have an error -- This needs to be refacorted because it is takign the average fore the entire row.
        dfPriceDiff = dfPriceDiff[dfPriceDiff.iloc[:,1:-4].mean(axis=1)<MAX_DIFF]
        dfPriceDiff = dfPriceDiff[dfPriceDiff.iloc[:,1:-4].mean(axis=1)>MIN_DIFF]

        #export the data as a csv file 
        dfPriceDiff.to_csv(USERFOLDER + '\PerfNoNopa_GUI\CLEANED_DATA' + FILENAME)


def CreateSQLDateStrings(SDATE, EDATE):

    SDlist = SDATE.split("-")
    EDlist = EDATE.split("-")

    date1 = datetime.date(int(SDlist[0]), int(SDlist[1]), int(SDlist[2]))
    date2 = datetime.date(int(EDlist[0]), int(EDlist[1]), int(EDlist[2]))
    day = datetime.timedelta(days=1)

    PivotString = ""
    AvgString = ""

    while date1 <= date2:
        PivotString = PivotString + "'" + date1.strftime('%Y-%m-%d') + "', "
        AvgString = AvgString + "\"'" + date1.strftime('%Y-%m-%d') + "'\"" + " + "
        date1 = date1 + day

    return PivotString[:-2], AvgString[:-2]

def CreateSnowflakeQueries(SDATE,EDATE):

    PivotString, AvgString = CreateSQLDateStrings(SDATE,EDATE)
    #QueryList = ['DELETE_TABLE_DATA', 'INSERT_TABLE_DATA', 'OVERLAPPING_0_1', 'MKDN_PRICE', 'MKDN_QTY_0_1', 'RETAIL_PRICE', 'MKDN_QTY']
    QueryList = ['OVERLAPPING_0_1', 'MKDN_PRICE', 'MKDN_QTY_0_1', 'RETAIL_PRICE', 'MKDN_QTY']
    #QueryList = ['MKDN_QTY_0_1']
    sqlList = []
    for query in QueryList: 
        filelocation = "/" + query + ".sql"
        text_file = open(filelocation, "r")
        sqlString = text_file.read()
        sqlString = sqlString.replace("(&SDATE)",SDATE)
        sqlString = sqlString.replace("(&EDATE)",EDATE)
        sqlString = sqlString.replace("(&PIVOT_STRING)", str(PivotString))
        sqlString = sqlString.replace("(&AVG_STRING)", str(AvgString))
        #print(sqlString)
        sqlList.append(sqlString)
    
    #write the strings to a file location
    with open('E.txt', 'w') as f:
        for line in sqlList:
            f.write(line)
            f.write('\n\n\n\n')
    f.close()


    return sqlList

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////TESTING SECTION FOR PROGRAM/////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def OpenCSVTESTING():
    data = pd.read_csv (USERFOLDER +'/PerfNoNopa_GUI/CSV_FILES/0_1.csv')
    dfOFFER = pd.DataFrame(data.iloc[:, 5:])
    dfOFFER = dfOFFER.fillna(0)

    #REFORMATTING ROWS
    dfOFFER_TEST = pd.DataFrame(data.iloc[:, 3])

    #4 variables at the front of the dataframes
    dfIndex = pd.DataFrame(data.iloc[:, 1:5])

    data = pd.read_csv (USERFOLDER +'/PerfNoNopa_GUI/CSV_FILES/MKDN_PRICE.csv')

    #TESTING REFORMATTING ROWS
    dfMKDN = pd.DataFrame(data)
    dfTEMP = dfMKDN[~dfMKDN.cic.isin(dfOFFER_TEST.cic)]
    RemoveCICS = dfTEMP['cic'].values.tolist()
    dfMKDN = dfMKDN.set_index('cic')
    dfMKDN.drop(RemoveCICS, inplace = True)
    dfMKDN = dfMKDN.reset_index()
    dfMKDN = dfMKDN.iloc[:,5:]
    dfMKDN = dfMKDN.fillna(0)

    data = pd.read_csv (USERFOLDER +'/PerfNoNopa_GUI/CSV_FILES/RETAIL_PRICE.csv')

    dfRETAIL = pd.DataFrame(data)
    dfRETAIL = dfRETAIL.set_index('cic')
    dfRETAIL.drop(RemoveCICS, inplace = True)
    dfRETAIL = dfRETAIL.reset_index()
    dfRETAIL = dfRETAIL.iloc[:, 5:]
    dfRETAIL = dfRETAIL.fillna(0)

    data = pd.read_csv (USERFOLDER +'/PerfNoNopa_GUI/CSV_FILES/MKDN_QTY_0_1.csv')

    dfQTY = pd.DataFrame(data)
    dfQTY = dfQTY.set_index('cic')
    dfQTY.drop(RemoveCICS, inplace = True)
    dfQTY=dfQTY.reset_index()
    dfQTY = dfQTY.iloc[:, 5:]
    dfQTY = dfQTY.fillna(0)

    data = pd.read_csv (USERFOLDER +'/PerfNoNopa_GUI/CSV_FILES/MKDN_QTY.csv')

    dfIQTY  = pd.DataFrame(data)
    dfIQTY = dfIQTY.set_index('cic')
    dfIQTY.drop(RemoveCICS, inplace = True)
    dfIQTY = dfIQTY.reset_index()
    dfIQTY  = dfIQTY.iloc[:, 5:]
    dfIQTY  = dfIQTY.fillna(0)
    
    #sum the rows of the dataframe so that they can be added to the dataframe
    dfIQTY = dfIQTY.mul(dfQTY)
    dfIQTY = dfIQTY.mul(dfOFFER)
    dfIQTY  = dfIQTY.fillna(0)

    #Eliminate offers and areas >10 from dataframes
    dfMKDN = dfMKDN.mul(dfOFFER)        
    dfRETAIL = dfRETAIL.mul(dfOFFER)
    dfMKDN = dfMKDN.mul(dfQTY)        
    dfRETAIL = dfRETAIL.mul(dfQTY)

    #created price difference
    dfPriceDiff = dfMKDN.div(dfRETAIL)

    #round to two decimal poitns
    dfPriceDiff = dfPriceDiff.round(2)

    #reindex the dataframe
    dfPriceDiff['unit'] = dfIQTY.sum(axis=1)
    dfPriceDiff['info'] = dfIndex['cic'].map(str)
    dfPriceDiff['upc'] = dfIndex['upc'].map(str)
    dfPriceDiff['div'] = dfIndex['div'].map(str)
    dfPriceDiff['cic'] = dfIndex['cic'].map(str)
    dfPriceDiff['offer'] = dfIndex['offer'].map(str)
    dfPriceDiff = dfPriceDiff.set_index('info')

    #complete analysis of dataframe
    AnalyzeDataframeTESTING(dfPriceDiff)

def AnalyzeDataframeTESTING(dfPriceDiff):

    #remove rows above a certain average--- This might have an error -- This needs to be refacorted because it is takign the average fore the entire row.
    '''This is where the issue was occuring. When I was creating the averages I was not using the correct range of data. This needs to understood for all date ranges 
    Convert all of the 0 values to "" or null so that the average can be taken of the data. CONVERT 0 VALUES'''
    dfPriceDiff = dfPriceDiff[dfPriceDiff.iloc[:,1:-5].mean(axis=1)<0.9]
    dfPriceDiff = dfPriceDiff[dfPriceDiff.iloc[:,1:-5].mean(axis=1)>0.1]
    print(dfPriceDiff)
    #export the data as a csv file 
    dfPriceDiff.to_csv(USERFOLDER + '\PerfNoNopa_GUI\CLEANED_DATA\CLEANED_DATA.csv')
    
"""
AUTHOR:@Nicholas Ackerman
DATE: 03/15/2021
PURPOSE: The purpose of this function is to have the user decide which process they would like to complete
Would they like to complete the SingleLookup method or the Automated function which will run through all of
the divisions.
REFERENCES: AutomationFunction, SingleDIVFunction
"""
def main():
    #TESTING
    CreateSnowflakeQueries('2021-07-01', '2021-09-30')

    #You do not need this seciton if you are pulling a series of dates twice. Run this process once and then use the OpenCSV Testing below.
    ans = input("TESTING[1] OR PULL DATA[2]?")

    #time complexity start
    t0 = time.time()
    
    if ans == '1':
        OpenCSVTESTING() 
    elif ans == '2':
        print("Application is Opening")
        app = MyApp()
        app.title("PERFORMANCE NO NOPA")
        app.mainloop()
    else:
        print("Please enter either [1] or [2]")

    #END OF THE PROGRAM
    t1 = time.time()
    print("End of Program: \n Completed in {} Minutes".format((t1-t0)/60))

if __name__ == "__main__":
    main()
