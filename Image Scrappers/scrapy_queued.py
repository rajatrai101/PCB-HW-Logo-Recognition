import re
import requests
import os
from bs4 import BeautifulSoup
import csv
import urllib.request

class Scrapper:
    def __init__(self, Name):
        self.name = Name
        self.dict = dict()

    def createPath(self, filename):
        # Create target Directory if doesn't exist
        if not os.path.exists(filename):
            os.mkdir(filename)
            print("Directory " , filename ,  " Created ")
        else:    
            print("Directory ", filename, " already exists")

    def retrieveImgs(self):
        imgCount=0
        for imgs, _, folderName,_ in list(self.dict.values())[1:]:
            l=len(imgs)
            imgCount+=l
            if l:
                for i in range(l):
                    path = ".\\Logos\\" + folderName
                    if i > 0:
                        path+="-"+str(i)
                    self.createPath(path)
                    try:
                        urllib.request.urlretrieve(imgs[i], path + "\\" + str(i) + ".png")
                    except:
                        print("error with ",folderName)
                        continue
                    if len(os.listdir(path))==0:
                        print("error with ",folderName)
        print("Total Images count",imgCount)
        
            
    def saveTablesToCSV(self, table, i):
        if not i:
            headers = []
            for header in table.findAll('th'):
                headers+=[header.text[:-1]]
            self.dict['Header'] = headers
        for row in table.findAll('tr')[1:]:
            tds = row.find_all('td')
            if len(tds)==4:
                imgs = tds[0].find_all('a')
                col0 = [img['href'] for img in imgs]
                cols = []
                for ele in tds[1:]:
                    txt = re.sub(r"[\n\t<>?]*", "", ele.text).strip()
                    for f in ["http","home","bought","now","Now"]:
                        i = txt.find(f)
                        if i > -1:
                            txt=txt[:i]  
                    cols.append(txt)
                for pre in ['Previous products:Current products:', "Previous products: Current products:", "Past: ", "Current products:"]:
                    if cols[-1] == pre:
                        cols[-1] = ""
                        break
                    if cols[-1].startswith(pre):
                        cols[-1] = cols[-1][max(0, cols[-1].rfind(':') + 1) :].strip()
                        break
                cols.insert(0, col0)
                if cols[2] not in self.dict:
                    self.dict[cols[2]] = cols
                else:
                    self.dict[cols[2]][0]=list(set(cols[0]+self.dict[cols[2]][0]))
                    print("Already there:", cols, "\n\t\t", self.dict[cols[2]])
            else:
                print("ERROR EMPTY:", tds, "||", len(tds))

    def output(self):
        with open('.\\'+self.name+'.csv', 'w', newline='', encoding='utf8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(self.dict.values())
        print(len(self.dict.values())-1)

def getPage(url):
        print("Getting Page: ", url)
        r = requests.get(url)
        soup = BeautifulSoup(r.content, 'html.parser')
        return soup
###MAIN
pages = [
        "https://how-to.fandom.com/wiki/How_to_identify_integrated_circuit_(chip)_manufacturers_by_their_logos/A-E",
        "https://how-to.fandom.com/wiki/Howto_identify_integrated_circuit_(chip)_manufacturers_by_their_logos/F-J",
        "https://how-to.fandom.com/wiki/Howto_identify_integrated_circuit_(chip)_manufacturers_by_their_logos/K-O",
        "https://how-to.fandom.com/wiki/Howto_identify_integrated_circuit_(chip)_manufacturers_by_their_logos/P-T",
        "https://how-to.fandom.com/wiki/Howto_identify_integrated_circuit_(chip)_manufacturers_by_their_logos/U-Z"
        ]
s = Scrapper("Logos")
s.createPath(s.name)
for i,page in enumerate(pages):
    soup = getPage(page)
    tab = soup.find('table')
    s.saveTablesToCSV(tab, i)

s.retrieveImgs()
s.output()