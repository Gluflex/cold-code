import requests
import urllib
import json
from IPython.core.display import display, HTML, JSON
from types import SimpleNamespace
import logging
import socket
import sys


#loading the certitcates for sixt API
session = requests.session()
certificate_path = '.'
session.cert = (f'{certificate_path}/signed-certificate.pem', f'{certificate_path}/private-key.pem')

headers = {
    "content-type": "application/json",
    "accept": "application/json",
    "api-version": "2022-06-01"
}
r=None
def getstats():
    url = 'https://web.api.six-group.com/api/findata'




#Requesting the API for information about the stock, it is meant to be a wildcard but we need to connect the training set to stock names
    http_request = "https://web.api.six-group.com/api/findata/v1/searchInstruments?query=KO&size=1"

    r = session.get(http_request, headers=headers) #, verify='./six-certificate/certificate.pem')
    if str(r.status_code)[0] != "1":
        logging.debug(f"HTTP{r.status_code}: {r.content}")
    else:
        logging.debug(f"HTTP{r.status_code}: {json.dumps(json.loads(r.content), indent=2)}")

    response_dict = json.loads(r.text)

    stats = response_dict["data"]["searchInstruments"][0]["hit"]["issuer"]["sector"]
    stats = stats + response_dict["data"]["searchInstruments"][0]["hit"]["issuer"]['name']
    stats = stats + response_dict["data"]["searchInstruments"][0]["hit"]["mostLiquidListing"]["ticker"]
    print (stats)

    
    return stats
def getstockstat():
     #parsing the request file   
    http_request = "https://web.api.six-group.com/api/findata/v1/listings/marketData/eodTimeseries?scheme=TICKER_BC&ids=AAPL_67&from=2022-03-18"
    stockdata = session.get(http_request, headers=headers) #, verify='./six-certificate/certificate.pem')
    if str(stockdata.status_code)[0] != "1":
        logging.debug(f"HTTP{stockdata.status_code}: {stockdata.content}")
    else:
        logging.debug(f"HTTP{r.status_code}: {json.dumps(json.loads(r.content), indent=2)}")
    
    #more parsing
    stockdata = json.loads(stockdata.text)
    eod_timeseries = stockdata['data']['listings'][0]['marketData']['eodTimeseries']
    first_value = eod_timeseries[0]['open']
    last_value = eod_timeseries[-1]['close']
    
    print("First value:", first_value)
    print("Last value:", last_value)
    
    if (first_value<last_value):
        return "higher than last week"
    else:
        return "lower than last week"
  #Using a general API because SIXT were missing the information we were trying to get  
def getdes():
    API_KEY = 'your_api_key_here'
    ticker_symbol = 'KO'

    url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker_symbol}&apikey={API_KEY}'

    response = requests.get(url)
    data = response.json()

    company_description = data['Description']
    print(company_description)
    
    return company_description

# sending data to the AR glasses
def tcpclient(data):
    
    TCP_IP = '192.168.50.69'
    TCP_PORT = 10008
    BUFFER_SIZE = 1024
    MESSAGE = datastream
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP, TCP_PORT))
    
    s.send(MESSAGE.encode())
    data = s.recv(BUFFER_SIZE)
    s.close()


            
datastream=str(getstats())+' , '+ str(getstockstat())+" , " + str(getdes())   
tcpclient(datastream)
        
    
    
   
