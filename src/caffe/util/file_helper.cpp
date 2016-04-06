#include <errno.h>
#include <iostream>
#include "caffe/util/file_helper.hpp"

bool CheckOutputDir(const std::string &path)
{
	if (IsDir(path))
		return true;
	if (path.empty())
		return false;

	size_t pos = path.find_last_of(Seperator());
	if (std::string::npos == pos)
		return false;
	std::string subpath = path.substr(0, pos);

	if (CheckOutputDir(subpath))
	{
#ifndef _WIN32
		if (mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0)
#else
		if (_mkdir(path.c_str()) != 0)
#endif
        {
            std::cout << "[ERROR]: Failed to mkdir " << path << std::endl;
            return false;
        }
	}

	return true;
}

#ifndef _WIN32
int GetSubFolders(const std::string &root, std::vector<std::string> &folders, const bool full)
{
    folders.clear();

    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(root.c_str())) == NULL)
    {
        return -1;
    }

    while ((dirp = readdir(dp)) != NULL) 
    {
        const std::string name(dirp->d_name);
        const std::string path = root + Seperator() + name;

        //check whether is directory
        struct stat statbuf;
        stat(path.c_str(), &statbuf);
        if(!S_ISDIR(statbuf.st_mode))
            continue;

        //default directory names
        if(name[0] == '.')
            continue;

        if (full)
            folders.push_back(path);
        else
            folders.push_back(name);
    }
    closedir(dp);

    return 0;
}
#else
int GetSubFolders(const std::string &root, std::vector<std::string> &folders, const bool full)
{
	folders.clear();

	if (root.empty())
		return -1;
	if (!IsDir(root))
		return -1;	

	WIN32_FIND_DATAA wfData;
	char filter[MAX_PATH];
	sprintf_s(filter, "%s\\*", root.c_str());
	HANDLE hFind = FindFirstFileA(filter, &wfData);

	if (INVALID_HANDLE_VALUE == hFind)	
	{
		return -1;
	}
	else
	{
		BOOL fnext = TRUE;
		while (fnext)
		{
			if (FILE_ATTRIBUTE_DIRECTORY == (wfData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
			{
				std::string tmp(wfData.cFileName);

				if (0 != tmp.compare(".") && 0 != tmp.compare(".."))
				{
					if (full)
						folders.push_back(root + "\\" + tmp);
					else
						folders.push_back(tmp);
				}
			}
			fnext = FindNextFileA(hFind, &wfData);
		}
		FindClose(hFind);
	}
	return 0;
}
#endif

int GetAllChildFolders(const std::string &root, std::vector<std::string> &pathes)
{
    //width-first-search
    pathes.clear();
    pathes.push_back(root);
    int p = 0;
    while(p < pathes.size())
    {
        const std::string &path_cur = pathes[p];
        std::vector<std::string> folders_cur;
        GetSubFolders(path_cur, folders_cur, true);

        if (!folders_cur.empty())
        {
            pathes.insert(pathes.end(), folders_cur.begin(), folders_cur.end());
        }
        p++;
    }

    return 0;
}

#ifndef _WIN32
int GetFilesInDir(const std::string &folder, const std::string &suffix, std::vector<std::string> &names, const bool full)
{
    names.clear();

    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(folder.c_str())) == NULL)
    {
        std::cout << "Error(" << errno << ") opening " << folder << std::endl;
        exit(-1);
    }

    const size_t len_suffix = suffix.size();
    while ((dirp = readdir(dp)) != NULL) 
    {
        const std::string name(dirp->d_name);

        //default directory names
        if(name[0] == '.')
            continue;

        const size_t len_cur = name.size();
        if (len_cur < len_suffix + 1)
            continue;

        if (name.substr(len_cur - len_suffix, len_suffix) != suffix)
            continue;

        if (full)
            names.push_back(folder + Seperator() + name);
        else
            names.push_back(name);
    }
    closedir(dp);

    return 0;
}
#else
int GetFilesInDir(const std::string &folder, const std::string &suffix, std::vector<std::string> &names, const bool full)
{
	HANDLE hFind;
	WIN32_FIND_DATAA fileData;
	char filter[MAX_PATH];
	sprintf_s(filter, "%s\\*%s", folder.c_str(), suffix.c_str());
	hFind = FindFirstFileA(filter, &fileData);
	//FindNextFile(hFind, &fileData);
	if (INVALID_HANDLE_VALUE == hFind)
	{
		printf("no files %s\\*%s are found\n", folder.c_str(), suffix);
		return -1;
	}
	else
	{
		BOOL fNext = TRUE;
		while (fNext)
		{
			if (FILE_ATTRIBUTE_DIRECTORY == (fileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
			{
				fNext = FindNextFileA(hFind, &fileData);
				continue;
			}

			if (full)
				names.push_back(folder + Seperator() + fileData.cFileName);
			else
				names.push_back(fileData.cFileName);
			
			fNext = FindNextFileA(hFind, &fileData);
		}
		FindClose(hFind);
	}
	return 0;
}
#endif

#ifndef _WIN32
int CountFilesInDir(const std::string &folder, const std::string &suffix)
{
    int count = 0;

    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(folder.c_str())) == NULL)
    {
        std::cout << "Error(" << errno << ") opening " << folder << std::endl;
        exit(-1);
    }

    const size_t len_suffix = suffix.size();
    while ((dirp = readdir(dp)) != NULL) 
    {
        const std::string name(dirp->d_name);

        //default directory names
        if(name[0] == '.')
            continue;

        const size_t len_cur = name.size();
        if (len_cur < len_suffix + 1)
            continue;

        if (name.substr(len_cur - len_suffix, len_suffix) != suffix)
            continue;

        count++;
    }
    closedir(dp);

    return count;
}
#else
int CountFilesInDir(const std::string &folder, const std::string &suffix)
{
	int count = 0;

	HANDLE hFind;
	WIN32_FIND_DATAA fileData;
	char filter[MAX_PATH];
	sprintf_s(filter, "%s\\*%s", folder.c_str(), suffix.c_str());
	hFind = FindFirstFileA(filter, &fileData);
	//FindNextFile(hFind, &fileData);
	if (INVALID_HANDLE_VALUE == hFind)
	{
		printf("no files %s\\*%s are found\n", folder.c_str(), suffix);
	}
	else
	{
		BOOL fNext = TRUE;
		while (fNext)
		{
			if (FILE_ATTRIBUTE_DIRECTORY == (fileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
			{
				fNext = FindNextFileA(hFind, &fileData);
				continue;
			}
			count++;

			fNext = FindNextFileA(hFind, &fileData);
		}
		FindClose(hFind);
	}

	return count;
}
#endif

int GetAllFiles(const std::string &root, const std::string &suffix, std::vector<std::string> &pathes_file)
{
    std::vector<std::string> folders;
    GetAllChildFolders(root, folders);

    pathes_file.clear();
    for (int i = 0; i < int(folders.size()); ++i)
    {
        std::vector<std::string> pathes_cur;
        //if (GetFilesInDir(folders[i], suffix, pathes_cur, true) != 0)
        //    return -1;
		GetFilesInDir(folders[i], suffix, pathes_cur, true);
        pathes_file.insert(pathes_file.end(), pathes_cur.begin(), pathes_cur.end());
    }

    return 0;
}

void FileParts(const std::string &path, std::string &folder, std::string &name)
{
    const size_t pos = path.find_last_of("/\\");
    folder = path.substr(0, pos);
    name = path.substr(pos+1);
}

void FileParts(const std::string &path, std::string &folder, std::string &name, std::string &ext)
{
    const size_t pos = path.find_last_of("/\\");
    folder = path.substr(0, pos);
    name = path.substr(pos+1);

    const size_t pos_dot = name.find_last_of(".");
    ext = name.substr(pos_dot);
    name = name.substr(0, pos_dot);
}

#ifndef _WIN32
#include <sys/sendfile.h> 
#include <fcntl.h>         // open
#include <unistd.h>        // close

void CopyFile(const char *src, const char *dst)
{
    int source = open(src, O_RDONLY, 0);
    int dest = open(dst, O_WRONLY | O_CREAT, 0644);

    // struct required, rationale: function stat() exists also
    struct stat stat_source;
    fstat(source, &stat_source);

    sendfile(dest, source, 0, stat_source.st_size);

    close(source);
    close(dest);
}
#endif

#ifdef _WIN32
#include <windows.h>
#elif MACOS
#include <sys/param.h>
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif
 
int getNumberOfCores() {
#ifdef WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
#elif MACOS
    int nm[2];
    size_t len = 4;
    uint32_t count;
 
    nm[0] = CTL_HW; nm[1] = HW_AVAILCPU;
    sysctl(nm, 2, &count, &len, NULL, 0);
 
    if(count < 1) {
    nm[1] = HW_NCPU;
    sysctl(nm, 2, &count, &len, NULL, 0);
    if(count < 1) { count = 1; }
    }
    return count;
#else
    return sysconf(_SC_NPROCESSORS_ONLN);
#endif
}
