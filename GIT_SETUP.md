# Setting Up Git Repository - Step-by-Step Guide

## Prerequisites

Before you begin, ensure you have Git installed:

### Windows
```bash
# Check if Git is installed
git --version

# If not installed, download from: https://git-scm.com/download/win
# Or use chocolatey: choco install git
```

### macOS
```bash
# Using Homebrew
brew install git
```

### Linux
```bash
# Ubuntu/Debian
sudo apt-get install git

# Fedora
sudo dnf install git
```

---

## Step 1: Initialize Local Repository

Open terminal/PowerShell in your project directory:

```bash
cd "D:\Final Year Project"

# Initialize git repository
git init
```

You should see:
```
Initialized empty Git repository in D:/Final Year Project/.git/
```

---

## Step 2: Configure Git (First Time Only)

Set your identity (required):

```bash
# Set your name
git config --global user.name "Your Name"

# Set your email
git config --global user.email "your.email@example.com"

# Verify configuration
git config --list
```

---

## Step 3: Add All Files to Staging

```bash
# Add all files (respects .gitignore)
git add .

# Verify what will be committed
git status
```

Expected output:
```
On branch master

No commits yet

Changes to be committed:
  new file:   README.md
  new file:   requirements.txt
  new file:   biomedical_waste/
  ...
```

---

## Step 4: Create Initial Commit

```bash
# Commit with descriptive message
git commit -m "Initial commit: Biomedical Waste Detection System v1.0"
```

---

## Step 5: Create Remote Repository

Choose one of these options:

### Option A: GitHub
1. Go to https://github.com/new
2. Create repository named: `biomedical-waste-detection`
3. **Do NOT initialize** with README/license (you already have them)
4. Copy the repository URL (HTTPS or SSH)

### Option B: GitLab
1. Go to https://gitlab.com/projects/new
2. Create repository named: `biomedical-waste-detection`
3. Copy the repository URL

### Option C: Bitbucket
1. Go to https://bitbucket.org/repo/create
2. Create repository named: `biomedical-waste-detection`
3. Copy the repository URL

### Option D: Local Server
Skip this step - your local repo is sufficient

---

## Step 6: Add Remote (if using GitHub/GitLab/Bitbucket)

```bash
# Add remote repository
# Replace URL with your actual repository URL
git remote add origin https://github.com/YOUR_USERNAME/biomedical-waste-detection.git

# Verify remote is added
git remote -v
```

Expected output:
```
origin  https://github.com/YOUR_USERNAME/biomedical-waste-detection.git (fetch)
origin  https://github.com/YOUR_USERNAME/biomedical-waste-detection.git (push)
```

---

## Step 7: Push to Remote (if using GitHub/GitLab/Bitbucket)

```bash
# For first push, use -u to set upstream
git branch -M main

git push -u origin main
```

On Windows, you may be prompted for credentials:
- Use GitHub username for username
- Generate Personal Access Token (PAT) for password:
  - GitHub: https://github.com/settings/tokens
  - Click "Generate new token"
  - Select scopes: `repo`, `workflow`

---

## ✅ Verification

Check your repository was set up correctly:

```bash
# View commit history
git log

# View repository status
git status

# View remote configuration
git remote -v

# View branches
git branch -a
```

---

## 📁 What Gets Ignored

The `.gitignore` file automatically excludes:

```
✓ Virtual environment (venv/)
✓ Python cache (__pycache__/)
✓ IDE files (.vscode/, .idea/)
✓ OS files (.DS_Store, Thumbs.db)
✓ Temporary files (*.tmp, *.log)
✓ Large model weights (*.pt)
✓ Test outputs
✓ Environment files (.env)
```

Large files like model weights are ignored to keep repository size small.

---

## 🚀 Regular Git Workflow

### After Making Changes

```bash
# 1. See what changed
git status

# 2. Add changes
git add .

# 3. Commit with message
git commit -m "Description of changes"

# 4. Push to remote (if applicable)
git push origin main
```

### Example Workflow

```bash
# Fix dashboard bug
git add dashboard/streamlit_app.py
git commit -m "Fix: Dashboard display issues on small screens"
git push origin main

# Add new feature
git add biomedical_waste/hazard/scoring.py
git commit -m "Feature: Add confidence-weighted hazard scoring"
git push origin main

# Update documentation
git add README.md
git commit -m "Docs: Update deployment instructions"
git push origin main
```

---

## 🔄 Useful Git Commands

### Clone Repository (on another machine)

```bash
git clone https://github.com/YOUR_USERNAME/biomedical-waste-detection.git
cd biomedical-waste-detection
```

### Create New Branch (for features)

```bash
# Create and switch to new branch
git checkout -b feature/new-feature-name

# Make changes and commit
git add .
git commit -m "Add new feature"

# Push branch
git push -u origin feature/new-feature-name

# Create Pull Request on GitHub for review
```

### View Commit History

```bash
# View last 10 commits
git log --oneline -10

# View ALL commits
git log --oneline

# View specific file history
git log --oneline -- filename.py
```

### Undo Changes

```bash
# Discard changes in working directory
git checkout -- filename.py

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1
```

---

## 📋 Complete Quick Reference

```bash
# 1. Initialize
git init
git config --global user.name "Your Name"
git config --global user.email "email@example.com"

# 2. Initial commit
git add .
git commit -m "Initial commit: Biomedical Waste Detection System"

# 3. Add remote (GitHub example)
git remote add origin https://github.com/username/biomedical-waste-detection.git

# 4. Push
git branch -M main
git push -u origin main

# 5. Later commits
git add .
git commit -m "Description of changes"
git push
```

---

## 🔐 Security Best Practices

### 1. Never Commit Secrets
```bash
# Create .env file for secrets (add to .gitignore)
API_KEY=xxxx
DATABASE_PASSWORD=xxxx
```

### 2. Use SSH Keys (More Secure)
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Add to GitHub: https://github.com/settings/keys

# Use SSH URL instead of HTTPS
git remote set-url origin git@github.com:username/biomedical-waste-detection.git
```

### 3. Protect Main Branch (GitHub)
1. Go to repository Settings
2. Click "Branches"
3. Add branch protection rule for `main`
4. Require pull request review before merge

---

## 📊 Repository Structure After Setup

```
biomedical-waste-detection/
├── .git/                          # Hidden - Git metadata
├── .gitignore                     # This file
├── README.md                      # Documentation
├── requirements.txt               # Dependencies
├── biomedical_waste/              # Source code
├── dashboard/                     # Web interface
├── data/                          # Training data
├── runs/                          # Model outputs
│   └── detect/biomedical_waste_30ep/
│       └── weights/
│           └── .gitignore         # Ignores *.pt files
└── ... (other project files)
```

---

## 🆘 Troubleshooting

### Issue: "fatal: not a git repository"
**Solution**: Run `git init` in project directory

### Issue: "everything up-to-date" but nothing pushed
**Solution**: Check remote is added correctly
```bash
git remote -v
# Should show: origin  https://...
```

### Issue: Authentication failed
**Solution**: Use Personal Access Token (PAT)
1. Generate PAT on GitHub/GitLab
2. Use as password when prompted
3. Or use SSH keys (more secure)

### Issue: .gitignore not working
**Solution**: Remove cached files
```bash
git rm -r --cached .
git add .
git commit -m "Apply gitignore"
```

### Issue: Accidentally committed large files
**Solution**: Remove from history
```bash
git filter-branch --tree-filter 'rm -f large_file.pt' HEAD
git push -f origin main
```

---

## 📚 Additional Resources

- **Git Documentation**: https://git-scm.com/doc
- **GitHub Guides**: https://guides.github.com/
- **GitLab Docs**: https://docs.gitlab.com/
- **Bitbucket Documentation**: https://confluence.atlassian.com/bitbucket/

---

## ✅ Next Steps

After setting up Git:

1. ✅ Create `.gitignore` (already done)
2. ✅ Initialize local repo: `git init`
3. ✅ Configure Git: `git config`
4. ✅ Make initial commit: `git commit`
5. ✅ Create remote repository (GitHub/GitLab/etc)
6. ✅ Push to remote: `git push`
7. ✅ Set up branch protection (optional)
8. ✅ Invite collaborators (optional)

---

## 🎉 You're Ready!

Your repository is now set up and ready for version control. 

**Start making commits:**
```bash
# When you make changes
git add .
git commit -m "Description of your changes"
git push origin main
```

---

**Last Updated**: 2026-04-28  
**Project**: Biomedical Waste Detection System v1.0.0
