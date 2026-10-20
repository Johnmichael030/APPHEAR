from core import RUN,fetch,write
from html.parser import HTMLParser
import json

class Fields(HTMLParser):
 def __init__(self):super().__init__();self.forms=[];self.form=None;self.select=None;self.option=None;self.links=[];self.scripts=[]
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if tag=='form':self.form={'attributes':a,'inputs':[],'selects':[]};self.forms.append(self.form)
  if tag=='input' and self.form is not None:self.form['inputs'].append(a)
  if tag=='select':self.select={'attributes':a,'options':[]};self.form['selects'].append(self.select)
  if tag=='option':self.option={'attributes':a,'text':''};self.select['options'].append(self.option)
  if tag=='a':self.links.append(a)
  if tag=='script' and a.get('src'):self.scripts.append(a['src'])
 def handle_data(self,text):
  if self.option is not None:self.option['text']+=text.strip()
 def handle_endtag(self,tag):
  if tag=='option':self.option=None
  if tag=='select':self.select=None
  if tag=='form':self.form=None

def main():
 for url,name in [('https://dai.cs.rutgers.edu/asllvd/signbank/rit_sentence_signs_2025_11_01.csv','rit_sentence_signs_2025_11_01.csv'),
  ('https://dai.cs.rutgers.edu/dai/s/dai','dai.html'),('https://www.bu.edu/asllrp/about-datasets.pdf','about-datasets.pdf'),
  ('https://www.bu.edu/asllrp/signbank-terms.pdf','signbank-terms.pdf')]:
  try:fetch(url,name)
  except Exception as e:print(json.dumps({'failed':name,'error':str(e)}),flush=True)
 page=RUN/'sources/dai.html'
 if page.exists():
  parser=Fields();parser.feed(page.read_text(encoding='utf-8',errors='replace'))
  write(RUN/'sources/dai_form_inventory.json',{'forms':parser.forms,'links':parser.links,'scripts':parser.scripts})
  print(json.dumps({'forms':parser.forms,'scripts':parser.scripts}),flush=True)

if __name__=='__main__':main()
