​​

# Welcome to the Chem160 code repository
- Overview
- Requirements
- Installation guide
- Working with the virtual machines
- Updating/Downloading homework
- Submitting homework
- Extra info

## Overview
In this course you'll get a sense of how quantum mechanics is used to solved chemistry problems. One of the ways we will get to understand the quantum world will be via code, we will be coding many of the ideas of the course and creating simulations that solve quantum phenomena.

We will primarily use Ipython notebooks, a great tool for sharing code, rapidly prototyping ideas, telling stories with multiple media.

For the first part of the course we will use common scientific python libraries that are common in many python installations.

Later we will use specialized modules and software that is normally used in a professional setting (industry, research, etc.). Since there is a wide variety of computer environments (Windows, MacOSX, Linux, 32-Bit and 64-Bit...etc), we have a created virtual machines. These machines are essentially self-contained operating systems which include all the software required for the course.

### First time programming?
No worries! This course is made for people of all skill levels. Having said that, the more you can program the more you can focus on learning the quantum chemistry part of it.

A great way to become highly proficient in python fast is taking the [python code academy course](https://www.codecademy.com/en/tracks/python). High schoolers typically finish the course in 13 hours (average).

## Requirements
### Essentials:
- Internet Browser, **Google Chrome** highly recommended.
- A terminal with ssh (MacOSX and Linux have native _good_ terminals, for Windows [cmder](http://cmder.net/) is recommended)
- [VirtualBox **5.X**](https://www.virtualbox.org/wiki/Downloads) with the Extension pack.
*   [VMD]([http://www.ks.uiuc.edu/Research/vmd/](http://www.ks.uiuc.edu/Research/vmd/)) or [Avogadro]([http://avogadro.cc/wiki/Main_Page](http://avogadro.cc/wiki/Main_Page)) if you want to visualize molecules.

- A sense of scientific adventure!

### Behind the hood:
**Note:** You do not need to install anything of this, it is all included in the virtual machines.

- [Python **2.7.x**](https://www.python.org/downloads/)
- [Ipython notebook](http://ipython.org/notebook.html) > **3.x** , we will use **4.x** also known as [Jupyter](https://jupyter.org)
- Modules: **[scipy](http://www.scipy.org/), [numpy](http://www.numpy.org/), [matplotlib](http://matplotlib.org/).**
- Chemistry modules: **[rdkit](http://www.rdkit.org/), [openbabel](http://openbabel.org/), [imolecule](http://patrick-fuller.com/imolecule/), [PyQuante](http://pyquante.sourceforge.net/), [OpenMM](http://openmm.org/), [chelabview](http://chemlab.github.io/chemlab/), [mdtraj](http://mdtraj.org/latest/).**

## Installation guide
Installing all the required software can be hard if you have never compiled code, hence we will rely on the virtual machines for any code related activities.

If *you are interested in setting up software for your computer we can offer guidance *, <font color="red"> but we will only offer dedicated support to VirtualBox problems.</font>

Remember we need three main components:

### 1 - Install VirtualBox version 5.x
##### (click me)

[![Virtual Box](https://box.scotch.io/banner-virtual-box.jpg)](https://www.virtualbox.org/)

#### Don't forget the VirtualBox Extensions Pack (it's on the same page)
Internet Browser, Google Chrome highly recommended. A terminal (MacOSX and Linux have native good terminals, for Windows cmder is recommended) VirtualBox 5.X with the Extension pack. A sense of scientific adventure!

### 2 - Internet Browser : Google-chrome
[![google-chrome](http://i.imgur.com/4063GcI.png)](http://www.google.com/chrome/)

### 3 - Terminal with ssh capabilities
We will be using the terminal to connect to our virtual machines remotely, this is where the ssh comes in.

#### Linux
A terminal is always installed in all Linux distributions, so should be easy to be found.

#### MacOSX
The default app in MacOSX is "Terminal", you can find it via the spotlight bar or within Apllications/Utilities.

#### Windows
Windows comes with two applications for command line purposes **Command Prompt** and **PowerShell**. Sometimes they do have ssh. A sure option is **Git Bash for Windows**, downloadable here:

[![Git Bash](https://git-for-windows.github.io/img/gwindows_logo.png**)](https://git-for-windows.github.io/)

### Does it have ssh?
Once open, run the following command:

```bash
ssh -V
```

should give information on the ssh version, something like this:

![](http://i.imgur.com/k4lY9Cu.png)

### Never used a Terminal?
No worries, check out this [great introductory blog post.](https://www.codecademy.com/blog/72-getting-comfortable-in-the-terminal-linux) and if you want to be a pro really fast check out [the code academy course](https://www.codecademy.com/en/courses/learn-the-command-line/lessons/navigation/exercises/your-first-command?action=resume).

### 4 - Copy of the virtual machine
There are two available flavors:
#### Flavor 1: 32-bit, no graphics, 6 GBs of storage
The virtual machine is a [Minimal ubuntu](https://help.ubuntu.com/community/Installation/MinimalCD) 15.04 disk image, loaded with the bare minimal necesities to function as a server plus course software. By default it is setup to use 768 Mbs of RAM and 12 MBs of video memory.

[Download the 32-bit version here.](https://mega.nz/#!jhUF1RKD!KqNvb-ha-jGIgns8p_Z3HqdJBRrP46AAAGoAJ1NNSO0)
##### Port number is <font color="green"> 3031</font>.

#### Flavor 2: 64-bit, graphics, 15 GBs of storage
The virtual machine is a [Lubuntu](http://lubuntu.net/) 15.04 disk image, loaded with a bare minimum graphical interface plus course software. By default it is setup to use 2GBs of RAM and 64 MBs of video memory.

[Download the 64-bit version here.](https://mega.nz/#!q4EjhAIT!OAC-vZHFb9W5StdFJ8Nn-cswUCkYqO8Jy2Akr3wKt90)
##### Port number is <font color="green"> 3032</font>.

##### The default password for each machine is <font size=6 color="green"> chem160</font>.

### 5 - Molecular Viewer

[![](http://www.ks.uiuc.edu/Research/vmd/images/vmd_logo_left.gif)](http://www.ks.uiuc.edu/Research/vmd/)

[![](https://upload.wikimedia.org/wikipedia/commons/c/c1/Avogadro.png)](http://avogadro.cc/wiki/Main_Page)



## Working with the virtual machines:
### *Step 1 :* Start up your virtualbox!
Duoble clink on the download .ova file and it will appear on your virtualbox screen, there just do start:

![](http://i.imgur.com/u3eTTpR.png)

After some loading you should arrive to this screen, with convenient shortcuts to our most useful programs:

![](http://imgur.com/tpWNP3x.png)

### *Step 2 :* Starting ipython notebook within your vbox
- Open a terminal by clicking on the terminal icon.
- Type **ipython notebook** and press enter, you will get some messages and the ipython notebook should start running.
- Open google-chrome.
- Type **localhost:8888** in the address bar.


You should see something like the following screen:

 ![](http://imgur.com/EfzyhMP.png)

Here you can navigate the files on the computer, you will be using the chem160 folder as a base for all code (problem sets, demos, extra stuff). You can enter a folder by clicking on it.

- Navigate to **chem160/extras/Test_Installation.ipynb**
- Run the code! This will test if you have all the required modules for the course.
- You can update this code repository by running the terminal command "**update-course**", so any time you need to download the latest problem sets, execute away!
- If by any chance you have problems updating... you can also delete the folder and re-run the command and it will download the files again.

### *Step 3 :* Headless Virtualbox (**recommended**\)
Virtual machines can be slow when running with graphical interfaces, it is possible to run the virtual machine in _headless_ mode, with means no graphical interface. Sort of like a calculation server. If you want to have super fast calculations, this is the way to go.

In this mode you will run the code on the virtualbox...but all graphics rendering and notebook editing will happen on your computer inside of a web browser.

#### *Step 3.1 :* Bootup your virtual machine in headless mode
![](http://imgur.com/181R2IL.png)

#### *Step 3.2 :* Open a terminal in your computer (not the virtual) and connect via ssh
![](http://imgur.com/VkzCGVz.png)

You can explore this windowless computer via ssh in a terminal window, using the following command:

```bash
ssh -p 3032 student@127.0.0.1
```
The **- p** part is for the port number so a **32 bit machine would be 3031.**

You might get a message asking if it is a secure connection, it is, type yes and enter. Remember the password for the machine is <font size=6 color="green">chem160</font>.If all was successful you will see a similar message:

![](http://imgur.com/7gXxuoC.png)

Now you have total control over the virtual machine...from there you can start an ipython server by typing

```bash
ipython-server
```

#### *Step 3.3 :* Open google-chrome (your computer) into localhost:8888
![](http://imgur.com/cUHPaL6.png)

__ Voila!__ Jupyter is alive! (remotely)

![](http://imgur.com/gs5te5J.png)

### Updating and starting problem sets
You can update the course material via:

```bash
update-course
```

__ Note: __ If by any chance you have problems updating... you can also delete the folder and re-run the command and it will download the files again. The command for deleting the folder (assuming you are in the home folder (**~**). **Be careful to not delete your work!**

```bash
rm -rf chem160
```
Once you have a problem set to work with, you should make a copy somewhere and work on that.

#### Alternative: Git clone the bitbucket repository
You can download all code related files by running the following command in a git enabled terminal:

```bash
git clone https://beangoben@bitbucket.org/beangoben/chem160.git
```
If you do not have git on your computer you can install it on your computer via:

[![Github](https://git-scm.com/images/logo@2x.png) ](https://git-scm.com/downloads)

### Submitting Homework
To submit your homework, create a folder named **lastname_firstinitial_hw#** and place your IPython notebooks, data files, and any other files in this folder.
For example Garry Kenny submitting problem set 3 would be **kenny_g_hw3**".

Your IPython Notebooks should be completely executed with the results visible in the notebook. We should not have to run any code. Compress the folder (please use .zip compression) and submit to the canvas site in the appropriate folder. **If we cannot access your work because these directions are not followed correctly, we will not grade your work.**

You can uploaded via the browser inside of the virtual machine or you could copy your files from the virtual machine  on to your normal computer.

One easy way to download your ipynb is via the browser, going to ** file > Download as > Ipython Notebook (.ipynb) ** such as:

![](http://i.imgur.com/kOHRW3M.png)

The file will appear in your typical browser download folder.

The command to copy a file or folder from Directory1 on the virtual box to Directory2 on your machine is, using the port number of you machine (3032 for 64 bit and 3031 for 32 bit):

```bash
scp -r -P 3032 student@127.0.0.1:Directory1 Directory2
```

For example to copy your problem set # 3 to your current directory would be:

```bash
scp -r -P 3032 student@127.0.0.1:~/chem160/problem_sets/3_problem_set .
```

or reverse!, to copy a file from your computer to the virtual box home directory:

```bash
scp -r -P 3032 file student@127.0.0.1:~/
```

## Extras
### ssh config file (Linux and MacOSX)
If you hate having to type -p 3032 student@127.0.0.1 everytime you can edit your ssh config file to make the command way shorter:

```bash
ssh chem160-box64
```

by inserting the following code into **~/.ssh/config**:

```bash
Host chem160-box64
    User student
    Hostname 127.0.0.1
    Port 3032
```

```bash
Host chem160-box32
    User student
    Hostname 127.0.0.1
    Port 3031
```
