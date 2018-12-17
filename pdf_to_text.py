import PyPDF2 as py
import numpy as np
import unicodedata
import string



file = open('', 'rb') #path to the pdf file
obj = py.PdfFileReader(file)
pages = obj.numPages

all_letters = string.ascii_letters + " "

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

st = ''

for i in range(0, pages-1):
    content = obj.getPage(i)
    try:
        content = content.extractText()
    except KeyError:
        continue
    content = content.split('  ')
    l = ''
    for k in content:
        if(len(k) > 50):
            l = k
            break
    l = l.lower()
    """
    All the below replacements are done in relation to the text pattern
    in orginal Harry Potter ebooks.  To process other text the replacements
    must be changed accordingly.
    """
    l = l.replace(chr(8217), "'")
    l = l.replace('-', '')
    l = l.replace('...', ' ')
    l = l.replace('mr.', 'mr')
    l = l.replace('mrs.', 'mrs')
    l = l.replace("'ll", " will")
    l = l.replace("'ve", " have")
    l = l.replace("'d", " would")
    l = l.replace("n't", " not")
    l = l.replace("'re", " are")
    l = l.replace("it's", "it is")
    l = l.replace("who's", "who is")
    l = l.replace("he's", "he is")
    l = l.replace("she's", "she is")
    l = l.replace("that's", "that is")
    l = l.replace("there's", "there is")
    l = l.replace("here's", "here is")
    l = l.replace("let's", "let us")
    l = l.replace("'s", "")
    l = unicodeToAscii(l)
    st += l


st = st.replace('   ', ' ')
st = st.replace('  ', ' ')

text = open('', 'a') # include the path of the file to which text is to written
text.write(st)
text.close()
