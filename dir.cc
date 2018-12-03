#include <string>
#include <iostream>
#include <dirent.h>

#define dir_info struct dir_info   \
{                                  \
                                   \
    char** files;                  \
    int filesAmount;               \
                                   \
}                                  \

int main(){
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir ("./inputs")) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL) {

            std::string str = ent->d_name;

            if(str.find(".png") != std::string::npos){

                printf ("%s\n", ent->d_name);

            }

            
        }
        
        closedir (dir);
        
    } else {
        
        /* could not open directory */
        perror ("");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}