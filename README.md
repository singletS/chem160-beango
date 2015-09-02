​​
# Welcome to the Chem160 code repository

*   Overview
*   Requirements
*   Installation guide
*   Working with the virtual machines
*   Updating/Downloading homework
*   Submitting homework
*   Extra info

## Overview

In this course you'll get a sense of how quantum mechanics is used to solved chemistry problems.
One of the ways we will get to understand the quantum world will be via code, we will be coding many of the ideas of the course and creating simulations that solve quantum phenomena.

We will primarily use Ipython notebooks, a great tool for sharing code, rapidly prototyping ideas, telling stories with multiple media.

For the first part of the course we will use common scientific python libraries that are common in many python installations.

Later we will use specialized modules and software that is normally used in a professional setting (industry, research, etc.). Since there is a wide variety of computer environments (Windows, MacOSX, Linux, 32-Bit and 64-Bit...etc), we have a created virtual machines. These machines are essentially self-contained operating systems which include all the software required for the course.

## Requirements

### Essentials:
* Internet Browser, **Google Chrome** highly recommended.
* A terminal with ssh (MacOSX and Linux have native *good* terminals, for Windows [cmder](http://cmder.net/) is recommended)
* [VirtualBox **5.X**](https://www.virtualbox.org/wiki/Downloads) with the Extension pack.
* A sense of scientific adventure!

### Behind the hood:
(Inside the virtual machines)

* [Python **2.7.x**](https://www.python.org/downloads/)
* [Ipython notebook](http://ipython.org/notebook.html) > **3.x** , we will use **4.x** also known as [Jupyter](https://jupyter.org)
* Modules: **[scipy](http://www.scipy.org/), [numpy](http://www.numpy.org/), [matplotlib](http://matplotlib.org/).**
* Chemistry modules: **[rdkit](http://www.rdkit.org/), [openbabel](http://openbabel.org/), [imolecule](http://patrick-fuller.com/imolecule/), [PyQuante](http://pyquante.sourceforge.net/), [OpenMM](http://openmm.org/).**

## Installation guide
Installing all the required software can be hard if you have never compiled code, hence we will rely on the virtual machines for any code related activities.

If **you are interested in setting up software for your computer we can offer guidance **, <font color="red"> but we will only offer dedicated support to VirtualBox problems.</font>

Remember we need three main components:

### **1** Install VirtualBox version 5.x
(click me)
[![Virtual Box](https://box.scotch.io/banner-virtual-box.jpg)](https://www.virtualbox.org/)

#### Don't forget the VirtualBox Extensions Pack (it's on the same page)

Internet Browser, Google Chrome highly recommended.
A terminal (MacOSX and Linux have native good terminals, for Windows cmder is recommended)
VirtualBox 5.X with the Extension pack.
A sense of scientific adventure!

### **2**  Internet Browser : Google-chrome

[![google-chrome](http://i.imgur.com/4063GcI.png)](http://www.google.com/chrome/)

### **3**  Terminal with ssh capabilities
We will be using the terminal to connect to our virtual machines remotely, this is where the ssh comes in.
![](http://i.imgur.com/FBcFMc8.png)

#### Linux  
A terminal is always installed in all Linux distributions, so should be easy to be found.
#### MacOSX
The default app in MacOSX is "Terminal", you can find it via the spotlight bar or within Apllications/Utilities.
#### Windows
Windows comes with two applications for command line purposes "Command Prompt" and "PowerShell". Sometimes they do have ssh. A sure option is "Git Bash for windows", downloadable here:

[![Git Bash](https://git-scm.com/images/logo@2x.png)](https://git-for-windows.github.io/)

### Does it have ssh?
Once open, run the following command:
```bash
ssh -V
```
should give information on the ssh version, something like this:
```bash
OpenSSH_6.7p1 Ubuntu-5ubuntu1.3, OpenSSL 1.0.1f 6 Jan 201
```

### **4**  Copy of the virtual machine
The virtual machine is a [Lubuntu](http://lubuntu.net/) 15.04 disk image, loaded with the bare minimum graphical interfaces plus course software.




## Working with the virtual machines:
![](extra/files/virtualbox.png)

### Start up your virtualbox!

### **3**  Starting ipython notebook within your vbox
It should look like the following image:
![](http://imgur.com/tpWNP3x.png)

From here:
* Open a terminal by clicking on the icon.
* Type the command **ipython notebook** and press enter, the ipython notebook should be running.
* Open google-chrome.
* Type **localhost:8888** in the address bar.
* You should see something like the following screen:
![](http://imgur.com/EfzyhMP.png)
Here you can navigate the files on the computer, you will be using the chem160 folder as a home base for all code (problem sets, demos, extra stuff).
You can enter a folder by clicking on it.

* Navigate to **chem160/extras/TestInstallation.ipynb**
* Run the code! This will test if you have all the required modules for the course.
* You can update this code repository by running "**update-course**", so any time you need to download the latest problem sets, execute away!

### **4** Headless Virtualbox (optional but **recommended**)
If you feel your virtual machine is a bit slow, it is possible to run the virtual machine in *headless* mode, with means no graphical interface. Sort of like a calculation server.

In this mode you will run the code on the virtualbox...but all graphics rendering and notebook editing will happen on your computer inside of a web browser.

#### 4.1 : Bootup your virtual machine in headless mode

![](http://imgur.com/181R2IL.png)
#### 4.2 : Open a terminal and connect via ssh

![](http://imgur.com/VkzCGVz.png)

You can explore this windowless computer via ssh in a terminal window, using the following command:

```bash
ssh -p 3032 student@127.0.0.1
```

If all was succesful you will see a message like this:
![](http://imgur.com/7gXxuoC.png)
Now you have total control over the virtual machine...from there you can start an ipython server by typing

```bash
ipython-server
```
#### 4.3 : Open a google-chrome into localhost:8888
![](http://imgur.com/cUHPaL6.png)

** Voila!** Jupyter is alive! (remotely)

![](http://imgur.com/gs5te5J.png)

### Updating/Downloading Homework

### Submitting Homework


You can update the course material via:
```bash
update-course
```
or copy a file or folder from Directory1 on the virtual box to directory2 on your machine
```bash
scp -r -p 3032 student@127.0.0.1:Directory1 Directory2
```
For example to copy your problem set # 1 to your current directory would be:

```bash
scp -r -p 3032 student@127.0.0.1:~/chem160/problem_sets/1_Intro_Ipython .
```
or reverse!, to copy a file on your computer to the virtual box:
```bash
scp -r -p 3032 file student@127.0.0.1:~/
```
## Extras
** ssh config file (Linux and MacOSX)**
If you hate having to type *-p 3032 student@127.0.0.1* everytime you can edit your ssh config file to make the command way shorter:
```bash
ssh chem160-box64
```
by inserting the following code into **!/.ssh/config**:

```bash
Host chem160-box64
    User student
    Hostname 127.0.0.1
    Port 3032
```