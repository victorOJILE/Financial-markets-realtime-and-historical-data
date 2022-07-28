const fs = require('fs');
const path = require('path');
console.log(path.resolve('public', 'chart.html'));
function readfile(myPath) {
    return file = fs.readFileSync(path.resolve('public', `${myPath}`), "utf-8", (err, data) => {
        if(err) { throw err; }else {
            return data;
        }
    });
}
exports.readfile = readfile;
// let buffer = new Uint8Array(Buffer.from('This is written from readFile.js'));
// fs.open('new-file.txt', (err, fd) => {
//     if(err) {
//         throw err;
//     }
//     fs.read(fd, (err, b, buff) => {
//         if(err) {
//             throw err;
//         }
//         fs.appendFile('new-file.txt', buff., 'utf8', (err) => {
//             if(err) {
//                 throw err;
//             }
//             console.log('appended');
//         });
//     });
//     fs.close(fd);
// });