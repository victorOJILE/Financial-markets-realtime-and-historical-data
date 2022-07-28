const http = require('http');
const readfile = require('./modules/readFile.js').readfile;
const PORT = process.env || 5000;
let html = readfile('chart.html');
let css = readfile("design.css");
let js = readfile("javascript.js");
let json = readfile("jsonFile.json");
const server = http.createServer((req, res) => { 
    console.log(req.url);
    if(req.url == "/") {
        res.setHeader("Content-Type", "text/html");
        res.end(html);
    }else if(req.url == '/design.css') {
        res.setHeader("Content-Type", "text/css");
        res.end(css);
    }else if(req.url == '/javascript.js') {
        res.setHeader("Content-Type", "application/javascript");
        res.end(js);
    }else if(req.url == '/timeseries.json') {
        res.setHeader("Content-Type", "application/json");
        res.end(json);
    }else {
        res.setHeader("Content-Type", "text/html");
        res.statusCode = 404;
        res.end('<h2>Page not found</h2>');
    }
});
server.listen(PORT);