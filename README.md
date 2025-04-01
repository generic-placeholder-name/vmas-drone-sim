# Drone Coverage Simmulation in VMAS

# Getting Started as a Contributor
Helpful video: https://youtu.be/MnUd31TvBoU?si=BMQ1mHRvrJd1we3S

To get started, clone this git repository. Within it, create a [virtual environment](https://realpython.com/python-virtual-environments-a-primer/). Once you're running the virtual environment, install the dependencies found in requirements.txt. For example, if using venv from the mac terminal,
```console
> cd vmas-drone-sim # change directory to the repo
> python -m venv venv # create a virtual environment (venv) in a folder called venv.
> source venv/bin/activate # activate virtual environment
> python -m pip install -r requirements.txt # install dependencies listed in requirements.txt
```

## Useful git commands
```console
## Create new branch
git clone [url]
git pull # Do while in main branch! Pulls changes from repo. Do this frequently and before creatinf new branch so working with latest code.
git branch [branch_name] # Create a new branch
git branch # shows what branch you're on
git switch [branch_name] #switch to new branch

## Stage files to be committed
git status # See changes you've made, and changes you've added, and changes that are uncommitted. Use frequently.
git add [file_name] # Stage changes you made to a spefific file.
git add --all # Stage all changes you made, preparing to be committed
git reset # undoes git add if you decide you don't want to stage something
git commit -m "[enter commit comment, maybe mention issue #[issuenum]]
git push origin [branch_name] #pushes changes to branch in remote repository so others can see

## merge changes to main
creat a pull request on GitHub so that the rest of the team can review your code
before merging to main branch.

```

A good practice is to create an issue for whatever you want to fix or work on, and then create a branch (from main) to work on that specific issue. Mention the issue number in the commit message and close the issue after you've merged to the main branch. Thoughtful comments are always helpful. Then delete your branch and begin the process again when you're ready to work on a new issue.

# Run Tests Before Merging to Main
Make sure that all tests in the tests folder run without failing. To do this, run the following command:
```
python -m pytest
```console