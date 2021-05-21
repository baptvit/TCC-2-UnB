[![Open Source ](https://badges.frapsoft.com/os/v1/open-source.png?v=103)](https://github.com/ellerbrock/open-source-badges/)

[![License: GPL V3](https://img.shields.io/badge/License-GPLv3-green.svg)](https://opensource.org/licenses/GPL-3.0)

# Contributing 
Hi, welcome to **Peregrine** project !

First of all, we thank you for your time and for your interest in contributing to this project.

Further below you can find all you need to know about this project and how you can be part of it.

Enjoy it !

---

# Table of contents 💬

[1. What should you know before you get started ?](one)

[`1.1 Who are we ?`](who-are-we)

<a href="./README.md">`1.2 What is this project about ?`</a>

<a href="why-contribute">`1.3. Why should I contribute ?`</a>

<a href="./CODE_OF_CONDUCT.md">`1.4 Basic principles - code of conduct`</a>

---

# <a href="one">1. What should I know before I get started ?</a>

## <a href="who-are-we">1.1 Who are we ?</a>

We take great interest in artificial inteligence and we are excited to help the worldwide comunity in any way. 

And, for this specific project, we'd like to make the difference by helping people grow their twitter accounts.

---

# First Contributions

It's hard. It's always hard the first time you do something. Especially when you are collaborating, making mistakes isn't a comfortable thing. We wanted to simplify the way new open-source contributors learn & contribute for the first time.

Reading articles & watching tutorials can help, but what's better than actually doing the stuff in a practice environment? This project aims at providing guidance & simplifying the way beginners make their first contribution. If you are looking to make your first contribution, follow the steps below.
<img align="right" width="300" src="https://raw.githubusercontent.com/deeplearningunb/pneumonia-diagnosis/master/assets/fork.png" alt="fork this repository" />

If you don't have git on your machine, [install it](https://help.github.com/articles/set-up-git/).

## Fork this repository

Fork this repository by clicking on the fork button on the top of this page.
This will create a copy of this repository in your account.

## Clone the repository

<img align="right" width="300" src="https://raw.githubusercontent.com/deeplearningunb/pneumonia-diagnosis/master/assets/clone.png" alt="clone this repository" />

Now clone the forked repository to your machine. Go to your GitHub account, open the forked repository, click on the clone button and then click the _copy to clipboard_ icon.

Open a terminal and run the following git command:

```
git clone "url you just copied"
```

where "url you just copied" (without the quote marks) is the url to this repository (your fork of this project). See the previous steps to obtain the url.

<img align="right" width="300" src="https://raw.githubusercontent.com/deeplearningunb/pneumonia-diagnosis/master/assets/copy-to-clipboard.png" alt="copy URL to clipboard" />

## Create a branch

Change to the repository directory on your computer (if you are not already there):

```
cd peregrine
```

Now create a branch using the `git checkout` command:

```
git checkout -b <add-your-new-branch-name>
```

(The name of the branch does not need to have the word _add_ in it, but it's a reasonable thing to include because the purpose of this branch, and the issue number

## Make necessary changes and commit those changes

Now open the files in a text editor and make the changes.

<img align="right" width="450" src="https://raw.githubusercontent.com/deeplearningunb/pneumonia-diagnosis/master/assets/git-status.png" alt="git status" />

If you go to the project directory and execute the command `git status`, you'll see there are changes.

Add those changes to the branch you just created using the `git add` command:

```
git add *
```

Now commit those changes using the `git commit` command:

```
git commit -m "Add Changes <your-changes>"
```

replacing `<your-changes>` with the answer to the question - What would change if this was approved?.

## Push changes to GitHub

Push your changes using the command `git push`:

```
git push origin <add-your-branch-name>
```

replacing `<add-your-branch-name>` with the name of the branch you created earlier.

## Submit your changes for review

If you go to your repository on GitHub, you'll see a `Compare & pull request` button. Click on that button.

<img style="float: right;" src="https://raw.githubusercontent.com/deeplearningunb/pneumonia-diagnosis/master/assets/compare-and-pull.png" alt="create a pull request" />

Now submit the pull request.

<img style="float: right;" src="assets/submit-pull-request.png" alt="submit pull request" />

Soon I'll be merging all your changes into the master branch of this project. You will get a notification email once the changes have been merged.

## Where to go from here 

Congrats!

You just completed the standard _fork -> clone -> edit -> PR_ workflow that you'll encounter often as a contributor!

## References

[1][contibuting to atom](https://github.com/atom/atom/blob/master/CONTRIBUTING.md#your-first-code-contribution) </br>
[2][adding a contributing file](https://help.github.com/en/github/building-a-strong-community/setting-guidelines-for-repository-contributors#adding-a-contributing-file) </br>
[3][First Contributions](https://github.com/firstcontributions/first-contributions/blob/master/README.md)
