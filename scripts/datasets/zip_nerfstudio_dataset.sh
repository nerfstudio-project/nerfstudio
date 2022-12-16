#!/bin/bash
# zip and copy to nerfstudio-static-server
for f in Egypt person kitchen plane dozer floating-tree aspen stump sculpture Giannini-Hall
do
    echo $f
    cd /home/ethanweber/nerfactory/data2/nerfstudio
    rm $f.zip
    zip -r $f.zip $f
    scp $f.zip nerfstudio-static-server:/mnt/disks/mountdir/nerfstudio/data/nerfstudio/
done