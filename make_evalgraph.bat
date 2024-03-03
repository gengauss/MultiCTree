@echo off

set root_path=%~dp0

set gpu=-1
set input_pathA=%root_path%\results\anterior\val\%%1fold_%3\
set input_pathP=%root_path%\results\posterior\val\%%1fold_%3\
set output_path=%root_path%\results\
set Group_num=%1_%3
set start=0
set end=99
set interval=1
set train=0

call python %root_path%\make_evalgraph_3fold_valonly.py -g %gpu% -inA %input_pathA% -inP %input_pathP% -o %output_path% -n %Group_num% -s %start% -e %end% -i %interval% -t %train%

pause