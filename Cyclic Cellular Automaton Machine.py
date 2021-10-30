import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import iterate_structure
import time
import random



class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        root.title("Cyclic Cellular Automaton Machine")
        root.resizable(width=False, height=False)
        self.x = 200
        self.y = self.x

        self.paused = True
        self.createWidgets() # δημιουργία γραφικού περιβάλλοντος
        self.init_automaton()


    def createWidgets(self):
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        self.ax.set_axis_off()
        self.fig.add_axes(self.ax)

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=1, rowspan=2)

        #CONTROLS
        controlsframe = tk.LabelFrame(root, text="Controls", font="14")
        controlsframe.grid(row=1, column=0)

        #play/pause
        self.pptext = tk.StringVar()
        self.pptext.set("pause")
        self.ppbt = tk.Button(
            master=controlsframe, state='disabled',width=14, textvariable=self.pptext, command=self.toggle_pause)
        self.ppbt.grid(row=0, column=0)

        #start
        self.startbutton = tk.Button(
            master=controlsframe, text="start", width=14, command=self.start_anim)
        self.startbutton.grid(row=1, column=0)

        #init
        self.initbutton = tk.Button(
            master=controlsframe, width=14, text="init automaton", command=self.init_automaton)
        self.initbutton.grid(row=2, column=0)


        #RULES
        rulesframe = tk.LabelFrame(root, text="Rules", font="14")
        rulesframe.grid(row=0,column=0, padx=5)

        bestframe = tk.Frame(rulesframe)
        bestframe.grid(row=0, column=0)

        #range
        self.Range = tk.IntVar()

        rangelabel = tk.Label(
            bestframe, text="Range:").grid(row=0, column=0, sticky=tk.W)
        rangebox = tk.Spinbox(bestframe, from_=1, to=256, width=5,
                            textvariable=self.Range).grid(row=0, column=1)

        #threshold
        self.Threshold = tk.IntVar()

        threslabel = tk.Label(
            bestframe, text="Treshold:").grid(row=1, column=0, sticky=tk.W)
        thresbox = tk.Spinbox(bestframe, from_=1, to=100000, width=5,
                              textvariable=self.Threshold).grid(row=1, column=1)

        #states
        self.States = tk.IntVar()

        stateslabel = tk.Label(
            bestframe, text="States:").grid(row=2, column=0, sticky=tk.W)
        statesbox = tk.Spinbox(bestframe, from_=2, to=100, width=5,
                               textvariable=self.States).grid(row=2, column=1)

        
        #neighborhood
        self.neiframe = tk.LabelFrame(rulesframe, text="Neighborhood:")
        self.neiframe.grid(row=1, column=0)
        self.Neighborhood = tk.IntVar()

        R1 = tk.Radiobutton(self.neiframe, text="Von Neuman",
                            variable=self.Neighborhood, value=1)
       
        R1.grid(row=0, column=0)

        R2 = tk.Radiobutton(self.neiframe, text="Moore",
                            variable=self.Neighborhood, value=2)
        R2.select()
        R2.grid(row=1, column=0)

        #random
        radombt = tk.Button(rulesframe,text="Random\nRules", command=self.randomgenerator).grid(row=4, column=0)

        #PRESETS
        presets_frame = tk.Frame(rulesframe)
        presets_frame.grid(row=0,column=2, rowspan=5)


        #presets
        tk.Label(presets_frame, text="Presets:").grid(row=0,column=0)

        self.presets_list = tk.Listbox(presets_frame, width= 30)
        self.presets_list.grid(row=1, column=0, columnspan=3, sticky=tk.EW)

        self.add_list()

        #load preset
        self.presets_list.bind("<<ListboxSelect>>", self.load_preset)


        #save
        tk.Button(presets_frame, text="save", command = self.save_rule).grid(row=2, column=0, sticky=tk.EW)
        self.save_name= tk.StringVar()
        self.save_entry = tk.Entry(presets_frame, textvariable= self.save_name, font="10")
        self.save_entry.grid(row=2, column=1, sticky=tk.NSEW)
        

    def add_list(self):
        #εισαγωγή λίστας προεπιλογών απο το αρχείο
        i=0
        self.presets_list.delete(0,'end')

        with open('presets.txt', mode='r') as data:
            for line in data:
                name, rules = line.split('=')

                rules = rules.strip('\n').split('/')

                if rules[3]=='N':
                    rules[3]=1
                else: rules[3]=2

                self.presets_list.insert(i, name)

                if i==0:
                    self.presets = rules
                else:
                    self.presets = np.vstack([self.presets, rules])

                i+=1


    def load_preset(self, event):
        #Φορτώνει τους κανόνες απο την επιλεγμένη προειλογή
        
        self.Range.set(
            self.presets[self.presets_list.curselection()[0]][0])

        self.Threshold.set(
            self.presets[self.presets_list.curselection()[0]][1])

        self.States.set(
            self.presets[self.presets_list.curselection()[0]][2])
        
        self.Neighborhood.set(
            self.presets[self.presets_list.curselection()[0]][3])


    def save_rule(self):
        #αποθηκεύεται ο συνδιασμός των κανόνων στο αρχείο προεπιλογών εφόσον έχει δοθεί όνομα απο τον χρήστη και ο συνδυασμός δεν είναι ήδη αποθηκευμένος 
        if (self.save_name.get()):
            range_ = str(self.Range.get())
            thres =  str(self.Threshold.get())
            states = str(self.States.get())
            nei = str(self.Neighborhood.get())

            rules = [range_, thres, states, nei]

            if (rules not in self.presets.tolist()): # ελεγχος υπαρξης της προεπιλογής
                with open('presets.txt', mode='a') as fl:
                    
                    if rules[3] == '1':
                        rules[3] = 'N'
                    else:
                        rules[3] = 'M'

                    fl.write("\n"+self.save_name.get()+"=" +
                            rules[0]+"/"+rules[1]+"/"+rules[2]+"/"+rules[3]) # αποθήκευση στο αρχείο
                    
                    self.save_entry.delete(0, 'end')

                self.add_list()
                self.save_entry.insert(0, "Added Rule")
            else:
                self.save_entry.insert(0, "Rule exist in presets")


    def start_anim(self):
        #ξεκινάει την αναπαραγωγή του αυτόματου
        self.animation = animation.FuncAnimation(
            self.fig, self.plot, interval=100, cache_frame_data=False, blit=True)

        self.paused = False

        self.ppbt["state"] = 'active'
        self.initbutton["state"] = 'active'
        self.startbutton["state"] = 'disabled'


    def randomgenerator(self):
        #δημιουργεί τυχαίους συνδιασμούς
        self.Neighborhood.set(random.randint(1,2))
        self.Range.set(random.randint(1,10))
        self.States.set(random.randint(2,20))

        
        foot = np.array(iterate_structure(
            generate_binary_structure(2, self.Neighborhood.get()), self.Range.get()), dtype=int)

        self.Threshold.set(random.randint(
            1, int(np.count_nonzero(foot == 1)/self.Range.get())))
        #το οριο κατοφλιού επιλέγεται να είναι ικρότερο απο το συνολικό πλήθος των γειτωνικών κελιών


    def init_automaton(self):
        #δημιουργία αυτόματου και αναπαράστασή του στο γραφικό περιβάλλον 

        if not self.paused:
            self.animation.event_source.stop()
            self.paused = True

        self.range = self.Range.get()
        self.threshold = self.Threshold.get()
        self.states = self.States.get()


        self.array = np.random.randint(0, self.states, (self.y, self.x))
        self.img = self.ax.pcolormesh(self.array,)# cmap='inferno')
        self.canvas.draw()

        self.ppbt["state"] = 'disabled'
        self.startbutton["state"] = 'active'

    
        self.foot = np.array(iterate_structure(
            generate_binary_structure(2, self.Neighborhood.get()), self.range), dtype=int)


    def toggle_pause(self):
        #έλεγχος παύσης/αναπαραγωγής
        if self.paused:
            self.animation.event_source.start()
            self.pptext.set("pause")
        else:
            self.animation.event_source.stop()
            self.pptext.set("play")
        self.paused = not self.paused


    def compute_func(self, values):
        #έλεγχος κανόνων για κάθε κελί
        cur = values[int(len(values)/2)]

        if cur == (self.states-1):
            count = np.count_nonzero(values == 0)
        else:
            count = np.count_nonzero(values == cur+1)
        
        if count >= self.threshold:
            cur += 1

        if cur == self.states:
            cur = 0

        return cur #returns curent cell's value


    def plot(self, i):
        #start = time.time()

        self.array = ndimage.generic_filter(
            self.array, self.compute_func, footprint=self.foot, mode='wrap') #εφαρμογή κανόνων για κάθε κεί του αυτόματου

        self.img.set_array(self.array) #αναβάθμιση απεικόνισης 

        #end = time.time()
        #print(end - start) χρησιμοποιήθηκαν για το υπολογισμό του μέσου χρόνου υπολογισμού 
        return self.img,


root = tk.Tk()
app = Application(master=root)
app.mainloop()
