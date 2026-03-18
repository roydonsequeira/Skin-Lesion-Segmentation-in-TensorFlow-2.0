# Commit & Push — Quick Guide

**When you change code and want to update GitHub:**

### 1. Go to your project folder
```powershell
cd "c:\Users\royse\Downloads\Skin-Lesion-Segmentation-in-TensorFlow-2.0-main"
```

### 2. See what changed (optional)
```powershell
git status
```

### 3. Stage your changes
```powershell
git add .
```

### 4. Commit with a message
```powershell
git commit -m "Update: describe what you changed"
```

### 5. Push to GitHub
```powershell
git push
```

---

**That’s it.** Same four steps every time: `git add .` → `git commit -m "message"` → `git push`.

---

### If something goes wrong

| Message | What to do |
|--------|------------|
| “nothing to commit, working tree clean” | No changes to save (or you already committed). |
| “nothing added to commit but untracked files present” | Run `git add .` again, then commit. |
| Push rejected / “fetch first” | Run `git pull`, then `git push`. |
