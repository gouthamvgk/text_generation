import PyPDF2 as py
import unicodedata
import string

fi = open(path1, 'rb') #path to the pdf file
le = open(path2, 'a') #path to the text file to be created

obj = py.PdfFileReader(fi)
pages = obj.numPages

all_letters = string.ascii_letters + " .,'"

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


for i in range(start_page, pages):
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
    l = unicodeToAscii(l)
    le.write(l)

le.close()
fi.close()
