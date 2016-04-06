#ifndef _FILE_HELPER_
#define _FILE_HELPER_

#ifdef _WIN32
	#include <Windows.h>
	#include <direct.h>
#else
	#include <dirent.h>
	#include <sys/stat.h>
	#include <sys/types.h>
#endif

#ifdef _MSC_VER
#pragma warning(disable:4996)
#endif

#include <vector>
#include <string>
#include <stdio.h>
#include <stdlib.h>

inline char Seperator()
{
	#ifdef _WIN32
		return '\\';
	#else
		return '/';
	#endif
}

#ifdef _WIN32
inline bool IsDir(const std::string &path)
{
	DWORD ftyp = GetFileAttributesA(path.c_str());
	if (ftyp == INVALID_FILE_ATTRIBUTES)
		return false;  //something is wrong with your path!

	if (ftyp & FILE_ATTRIBUTE_DIRECTORY)
		return true;   // this is a directory!

	return false;    // this is not a directory!
}
#else
inline bool IsDir(const std::string &path)
{
	DIR *dir = opendir(path.c_str());
	if (dir != NULL)
	{
		closedir(dir);
		return true;
	}
	else
		return false;
}
#endif

inline void CheckInputDir(const std::string &path)
{
	if (!IsDir(path))		
	{
		printf("%s does not exists\n", path.c_str());
		exit(-1);
	}
}

inline bool ExistFile(const std::string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }   
}

bool CheckOutputDir(const std::string &path);

inline std::string GetFileSuffix(const std::string &path)
{
    const size_t pos = path.find_last_of(".");
    return path.substr(pos+1);
}

inline std::string Fullfile(const std::string &folder, const std::string &name)
{
    if (folder[folder.size() - 1] == Seperator())
    {
        return folder + name;
    }
    else
        return folder + Seperator() + name;
}

int GetSubFolders(const std::string &root, std::vector<std::string> &folders, const bool full = false);

int GetAllChildFolders(const std::string &root, std::vector<std::string> &pathes);

int GetFilesInDir(const std::string &folder, const std::string &suffix, std::vector<std::string> &names, const bool full = false);

int CountFilesInDir(const std::string &folder, const std::string &suffix);

int GetAllFiles(const std::string &root, const std::string &suffix, std::vector<std::string> &pathes_file);

void FileParts(const std::string &path, std::string &folder, std::string &name);

void FileParts(const std::string &path, std::string &folder, std::string &name, std::string &ext);

#ifndef _WIN32
void CopyFile(const char *src, const char *dst);
#endif

int getNumberOfCores();
#endif //_FILE_HELPER
