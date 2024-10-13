# Git Prompt

### Dir Operation

mkdir : create a dir



cd : open a dir

cd .. : return to the previous dir



ls ： list all files and folders in a dir

> ls -a will also list the hidden files and folders



cat : view the content of the file



### Check Version

git -v : check the version of git in your computer



### Configure Users and Email Addresses

git config  --global user.name "JadeSprings"

git config  --global user.email 321xxxxxx23@zju.edu.cn

> --global means global configuration, which applies to all repo. If --global is omitted(i.e. --local), the configuration will only be valid for the current repo.
>
> --system means system-level configuration, it's valid for all users(Usually not used).



git config --global credential.helper store : save the username and password



git config --global --list : view the git's configuration



### Create Repo

##### Method 1

git init : create a repo directly in the local

> You can also specify a dir-name after the command. If you do, the repo will be created in the dir that you specified, whose root-dir is the current dir.



##### Methon 2

git clone : clone a repo from a remote server



### Working Area and File Status

There are 3 working areas in git : Working Directory(工作区, i.e.  current dir), Staging Area/Index(暂存区) and local repository(本地仓库, i.e.  .git).



git add : add the document from the working directory to staging area

git commit : commit the doucument from the staging area to local repository



![](C:\My\0ScientificReasearch\Notes\Git\img\1.png)





### Asscociate the local repo with the remote repo

git remote add <shortname> <url>

> shortname of the remote is usually "origin"
>
> git remote set-url origin https://github.com/JadeSprings/Notes.git ： change url

git branch -M main : specify the name of the branch as main

git push -u origin main : associate the local main branch with the main branch in the remote repo



git remote -v : check the shortname and address of the remote repo corresponding to the current repo



git pull origin <远程分支名>:<本地分支名> : pull the branch of the remote repo to the local and merge them



git fetch