import tkinter
import shutil
import tkinter.messagebox
import os
from stopmotion import stopmotion
x=0
customer=0
entry1=None
window=None

def windowsetting():
    global entry1
    global window
    window=tkinter.Tk()

    window.title("EatN")
    window.geometry("640x400+100+100")
    window.resizable(False, False)

    label1=tkinter.Label(window, text="당신이 몇 개를 가져갔는지, 우리는 알 수 밖에 없습니다.", width=50, height=3, relief="solid")
    label1.pack()

    button1 = tkinter.Button(window, text='실행', overrelief="solid", width=15, command=count, repeatdelay=1000, repeatinterval=100)
    button1.pack()

    label2=tkinter.Label(window, text="가져온 개수를 입력해주세요", width=50, height=3, relief="solid")
    label2.pack()
    entry1 = tkinter.Entry(window)
    entry1.pack()

    button2 = tkinter.Button(window, text = "클릭", command = determine)
    button2.pack()

    window.mainloop()

def count():
    global x
    x = stopmotion()

def determine():
    global customer
    if(x>int(entry1.get())): 
        response = tkinter.messagebox.askyesno("경고", "진짜로? 우리는 당신이 "+str(x)+"개 가져간 것으로 인식했습니다.")
        if response==1: 
            os.rename('SaveVideo.mp4', 'Warning_'+str(customer)+'.mp4')
            shutil.move('./Warning_'+str(customer)+'.mp4', './video')
            customer+=1
            tkinter.messagebox.showinfo("정보", "우리 매장 이용해주셔서 감사합니다.")
    else:
        os.rename('SaveVideo.mp4', 'MaybeSafe_'+str(customer)+'.mp4')
        shutil.move('./MaybeSafe_'+str(customer)+'.mp4', './video')
        tkinter.messagebox.showinfo("정보", "우리 매장 이용해주셔서 감사합니다.")
        customer+=1



windowsetting()
    
