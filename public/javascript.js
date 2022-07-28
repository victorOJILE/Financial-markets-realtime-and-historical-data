let elId = (id) => document.getElementById(id);
let elCls = (cls) => document.getElementsByClassName(cls);
let chartDiv = elId('overflown-div');
let html = document.documentElement;
function openFullScreen() {
	if(html.requestFullscreen) {
		html.requestFullscreen();
	}else if(html.webkitRequestFullscreen) {
		html.webkitRequestFullscreen();
	}else if(html.msRequestFullscreen) {
		html.msRequestFullscreen();
	}
}
function closeFullScreen() {
	if(html.exitFullscreen) {
		html.exitFullscreen();
	}else if(html.webkitExitFullscreen) {
		html.webkitExitFullscreen();
	}else if(html.msExitFullscreen) {
		html.msExitFullscreen();
	}
}

/* toggle timeframe buttons background color */
let tfToggle = {
	visible: elCls("visible")[0],
	tfBtns: elCls('tfBtn'),
	tf2: elCls('tfBtn')[1],
	indicateInitialTf() {
		this.tf2.style.backgroundColor = 'rgb(209, 213, 213)';
		this.tf2.style.color = 'black';
	},
	indicateTfChange() {
		let tis = this;
		return this.visible.addEventListener('click', function(event) {
			let target = event.target;
			for(let key of tis.tfBtns) {
				key.style.backgroundColor = 'rgb(78, 78, 78)'; 
				key.style.color = 'rgb(247, 247, 247)';
			}
			if(target.classList.contains('tfBtn')) {
				target.style.backgroundColor = 'rgb(209, 213, 213)';
				target.style.color = 'black';
				tf.innerText = target.innerText;
			}
		});
	}
}
tfToggle.indicateInitialTf();
tfToggle.indicateTfChange();

/* toggling timeframe buttons background ends here */

let svg = elId('chart');

let windowHeight = window.innerHeight;
window.addEventListener('resize', () => {
	let newHeight = window.innerHeight;
	if(newHeight - windowHeight > 100 || windowHeight - newHeight > 100) {
		windowHeight = newHeight;
		new CandleSticks().redrawCandles();
	}
});
let priceArray, len, drawgrid, candles;
let response = fetch('timeseries.json');
response.then(result => result.json())
.then(res => {
	priceArray = res;
	len = Object.entries(priceArray).length-1;
	/* new candle information at the top of the chart */
	newCandleInfo();
	/* Draw grid on chart */
	drawgrid = new GridClass().drawGrid();
	/* Draw candles */
	candles = new CandleSticks().drawCandles();
	fillPriceDiv();
	/* Enable mouse wheel horizontal scrolling */
	chartDiv.addEventListener('wheel', function(e) {
		if(e.deltaY >= 100) chartDiv.scrollLeft -= 50;
		if(e.deltaY <= -100) chartDiv.scrollLeft += 50;
	});
	
	tooltipFunc();

});
function newCandleInfo() {
	elId('open-price').innerText = priceArray[len].open;
	elId('highest-price').innerText  = priceArray[len].high;
	elId('lowest-price').innerText  = priceArray[len].low;
	elId('current-price').innerText  = priceArray[len].close;
}

/* Grid class for grid operations */
class GridClass {
	constructor() {
		this._x1 = 10; this._x2 = 10; this._y1 = 0; this._y2 = svg.height.animVal.value; 
		this._gridcolor = "rgb(150, 150, 150)";
	}
	drawGrid() {
		for(let i = 0; i < svg.width.animVal.value; i++) { // Draw vertical gridline until i variable is equal to svg width
			svg.innerHTML +=  `<g class="gridlines" stroke="${this._gridcolor}"><path stroke-width="0.3" stroke-dasharray="3, 3" d="M${this._x1} ${this._y1} L${this._x2} ${this._y2}" /></g>`;
			if(this._x1 > svg.width.animVal.value || this._x2 > svg.width.animVal.value) { break; }
			this._x1 += 17; this._x2 += 17;
		}
		this._x1 = 0; this._x2 = svg.width.animVal.value; this._y1 = 10; this._y2 = 10;
		for(let i = 0; i < svg.height.animVal.value; i++) { // Draw horizontal gridline until i variable is equal to svg width
			svg.innerHTML += `<g class="gridlines" stroke="rgb(150, 150, 150)"><path stroke-width="0.3" stroke-dasharray="3, 3" d="M${this._x1} ${this._y1} L${this._x2} ${this._y2}" /></g>`;
			if(this._y1 > svg.height.animVal.value || this._y2 > svg.height.animVal.value) { break; }
			this._y1 += 17; this._y2 += 17;
		}
	}
	gridColor(color = "rgb(150, 150, 150)") {
		let grids = elCls('gridlines');
		for(let grid of grids) {
			grid.setAttribute('stroke', color);
		}
		this._gridcolor = color;
	}
	removeGrid() {
		let grids = elCls('gridlines');
		for(let grid of grids) {
			if(grids.length > 0) {
				grid.remove();
				this.removeGrid();
			}
		}
	}
}

let high, low, px;
function color() {
	if(this.open > this.close) { return 'red'; }else { return 'white'; }
}
/* Draw, modify candlesticks */
let candlesX = [];
class CandleSticks {
	constructor() {
		this._high = 0; this._low = 0; this._count = true; this.time0Arr = []; this.time8Arr = [];
		this.time16Arr = [];
		this._x1 = 62.2; this._x2 = 60; this._id=0; this.bidlinecolor = "rgb(30, 30, 255)";  
	}
	drawCandles() {
		/*  Get highest and lowest point in price  */
		for(let i = 0; i< Object.keys(priceArray).length; i++) {
			let h = priceArray[i].high;
			let l = priceArray[i].low;
			if(h > this._high) this._high = h;
			if(this._count) {
				this._low = l; 
				this._count = false;
			}else {
				if(l < this._low) this._low = l;
			}
		}
		px = this._px = ((svg.height.animVal.value/1.4)/(this._high-this._low)).toFixed(3);
		high = this._high; low = this._low;
		/* Draw candlesticks */
		
		for(let i = 0; i< len+1; i++) {
			this._y1 = function() {
				if(priceArray[i].open < priceArray[i].close) {
					return (this._high - priceArray[i].close) * this._px+30;
				}else {
					return (this._high - priceArray[i].open) * this._px+30;
				}
			};
			this._y2 = function() {
				if(priceArray[i].open < priceArray[i].close) {
					let b = this._y1();
					return (((this._high - priceArray[i].open) * this._px) - b)+30;
				}else {
					return (((this._high - priceArray[i].close) * this._px) - this._y1())+30;
				}
			};
			
			this.wick = color.call(priceArray[i]); this.candleLine = this.wick;
			this.candlebody = this.wick;
			let candleType = (this.wick == "red")? "bear": "bull";
			let wickType = (this.wick == "red")? "bear": "bull";
			
			if(priceArray[i].time == '00:00') { // Draw period seperator
				svg.innerHTML += `<g stroke="rgb(255, 255, 255)"><path stroke-width="1" stroke-dasharray="3, 3" d="M${this._x1} ${0} L${this._x1} ${svg.height.animVal.value}" /></g>`;
				this.time0Arr.push(this._x1);
			}
			if(priceArray[i].time == '08:30') { this.time8Arr.push(this._x1); }
			if(priceArray[i].time == '16:30') { this.time16Arr.push(this._x1); }
			candlesX.push(Math.trunc(this._x1));
			svg.innerHTML += `<line title="Open: ${priceArray[i].open}
			High: ${priceArray[i].high}
						Low: ${priceArray[i].low}
						Close: ${priceArray[i].close}" x1="${this._x1}" y1="${(this._high - priceArray[i].high)* this._px+30}" x2="${this._x1}" y2="${(this._high - priceArray[i].low)*this._px+30}" style="stroke: ${this.wick}" class="${wickType}" />
						<rect x="${this._x2}" y="${this._y1()}" id="${this._id}" name="${candleType}" width="5" height="${this._y2()}" style="fill: ${this.candlebody};" />`;
						this._x1 = this._x1 + 8; this._x2 = this._x2 + 8; this._id++;
					}
					
					let svgWidth = svg.width.animVal.value, svgHeight = svg.height.animVal.value.toFixed(2);
					svg.innerHTML += `<g stroke="rgb(250, 250, 250)"><path stroke-width="1" d="M0 ${svgHeight} L${svg.width.animVal.value} ${svgHeight}" /></g>`;
					svg.innerHTML += `<g stroke-width="2"><path d="M0 ${svgHeight-25} L${svgWidth} ${svgHeight-25} L${svgWidth} ${svgHeight} L0 ${svgHeight}" style="fill: inherit" />
					<path stroke="white" d="M0 ${svgHeight -25} L${svgWidth} ${svgHeight-25}" />
		</g>`;		
		for(let i=0; i< this.time0Arr.length; i++) {
				svg.innerHTML += `<text x="${this.time0Arr[i]-14}" y="${svg.height.animVal.value.toFixed(2) - 12}" style="fill: white">00:00</text>`;
			}
			for(let i=0; i< this.time8Arr.length; i++) {
				svg.innerHTML += `<text x="${this.time8Arr[i] -14}" y="${svg.height.animVal.value.toFixed(2) - 12}" style="fill: white">08:30</text>`;
			}
			for(let i=0; i< this.time16Arr.length; i++) {
			svg.innerHTML += `<text x="${this.time16Arr[i] -14}" y="${svg.height.animVal.value.toFixed(2) - 12}" style="fill: white">16:30</text>`;	
		}
		chartDiv.scrollLeft = elId(`${this._id-1}`).x.animVal.value - document.body.offsetWidth/2;
		
		/* Draw bid line */
		svg.innerHTML += `<g class="bid-line" stroke="${this.bidlinecolor}"><path stroke-width="1" str d="M0 ${(this._high - priceArray[len].close)*this._px+30} L${svg.width.animVal.value} ${(this._high - priceArray[len].close)*this._px+30}" /></g>`;	
	}
	bidLineColor(color = "rgb(100, 100, 255)") {
		let bidline = elCls('bid-line')[0];
		bidline.setAttribute('stroke', color);
	}
	candleWick(type = 'bull', color = this.candlebody) {
		let wicks = elCls(type);
		for(let wick of wicks) {
			wick.style.stroke = color;
		}
	}
	candleBody(type = 'bull', bod = this.candlebody) {
		for(let i=0; i<len+1; i++) {
			let thisid = elId(i);
			if(thisid.getAttribute('name') == type) thisid.style.fill = bod;
		}
	}
	candleBorder(type = 'bull', border = this.candlebody) {
		for(let i=0; i<len+1; i++) {
			let thisid = elId(i);
			if(thisid.getAttribute('name') == type) thisid.style.stroke = border;
		}
	}
	wholeCandle(type = 'bull', color = 'red') {
		let wicks = elCls(type);
		for(let wick of wicks) {
			wick.style.stroke = color;
		}
		for(let i=0; i<len+1; i++) {
			let thisid = elId(i);
			if(thisid.getAttribute('name') == type) {
				thisid.style.stroke = color;
				thisid.style.fill = color;
			}
		}
	}
	redrawCandles() {
		svg.innerHTML = '';
		new GridClass().drawGrid();
		new CandleSticks().drawCandles();
		svg2.innerHTML = '';
		fillPriceDiv();
	}
}

let svg2 = elId('price-div');

/* Fill price div */

function fillPriceDiv() {
	high = Math.ceil(high);
	low = Math.floor(low);
	let v = high;
	let difference = (v-low)/8;
	// let dif = v-difference;
	let f = high;
	svg2.innerHTML += `<rect x="0" y="${(v-priceArray[len].close)*px+20}" id="current-price-rect" width="100%" height="14" style="fill: white;" />`;
	svg2.innerHTML +=  `<text id="close-price" x="0" y="${(v-priceArray[len].close)*px+30}" style="fill: black"> -${priceArray[len].close}</text>`;
	
	let px2 = ((svg2.height.animVal.value/1.4)/(v-low)).toFixed(3);
	
	for(let i=0; i<9; i++) {
		svg2.innerHTML += `<text x="0" y="${(f-priceArray[len].high)*px2+30}" style="fill: white"> -${v}</text>`;
		f = f+difference;  v = v-difference;
	}
}


/* Display tooltip with candle information */

let title = elCls('title')[0];
let target2 = document.body;
let x, y;
let rects = document.getElementsByTagName('rect');
let flag = false;
function tooltipFunc() {
	for(let i=0; i<rects.length; i++) {
		rects[i].addEventListener('mouseenter', function(e) {
			if(flag) return;
			let target = e.target;
			let width = window.innerWidth, clientx = e.clientX;
			x = (width - clientx < 130)? clientx -125: clientx; 
			y = e.clientY;
			title.style.top = y+"px" ; title.style.left = x+"px";
			title.innerText = `Open: ${priceArray[target.id].open}\nHigh: ${priceArray[target.id].high}\nLow: ${priceArray[target.id].low}\nClose: ${priceArray[target.id].close}\nTime: ${priceArray[target.id].fullTime}`;	
		setTimeout(() => {
			if(e.target == this) {
				title.classList.add('show');
			}
			target2 = target;
			flag = true;
		}, 1000);
	});
}	
}

target2.addEventListener('mouseout', function() {
	title.classList.remove('show');
	flag = false;
});

/* Display tooltip with candle information ends here */

/* Toggle toolBtns background color */

let scrollDiv = elCls('scrollDiv')[0];
let toolBtn = elCls('toolBtn');
scrollDiv.addEventListener('click', function(e) {
	let targ = e.target;
	for(let key of toolBtn) {
		key.classList.remove('active-toolBtn');
	}
	if(targ.classList.contains('toolBtn')) {
		targ.classList.add('active-toolBtn');
	}
	if(targ.parentNode.classList.contains('toolBtn')) {
		targ.parentNode.classList.add('active-toolBtn');
	}
	let d = false;
	for(let i = 0; i<toolBtn.length; i++) {
		if(toolBtn[i].classList.contains('active-toolBtn')) {
			d = true;	
	    }
	}
	if(d == false) toolBtn[0].classList.add('active-toolBtn');
});

let cursor = elId('cursor');
let objects = [];

/*  Crosshair  */
let crosshair = elId('crosshair');
let ch, crosshairid = 'crosshairy', vt, vtid= 'crosshairv'; 
let chartMainDiv = elCls('chartMainDiv')[0];

function drawCrosshair(e) {
	if(crosshair.classList.contains('active-toolBtn')) {
		vt = chartDiv.scrollLeft + e.clientX; ch = e.clientY;//-42;
		svg.innerHTML += `<g stroke="rgb(255, 255, 255)" class="line"><path id="${vtid}" stroke-width="1" d="M${vt} 0 L${vt} ${svg.height.animVal.value}" /></g>`;
		svg.innerHTML += `<g stroke="rgb(255, 255, 255)" class="line"><path id="${crosshairid}" stroke-width="1" d="M0 ${ch} L${ svg.width.animVal.value} ${ch}" /></g>`;
		ch=''; vt='';
	    svg.addEventListener('pointermove', pointermov);
	}
}
function pointermov(e) {
	svg.removeEventListener('pointerenter', drawCrosshair);
	vt = chartDiv.scrollLeft + e.clientX; ch = e.clientY-40;
	elId('crosshairy').setAttribute('d', `M${vt} 0 L${vt} ${svg.height.animVal.value - 25}`);
	elId('crosshairv').setAttribute('d', `M0 ${ch} L${ svg.width.animVal.value} ${ch}`);
	if(candlesX.indexOf(vt) != -1) {
		let index = candlesX.indexOf(vt);
				let timeMove = elId('timeMove');
				if(timeMove) { 
					timeMove.setAttribute('x', vt-40);
					timeMove.setAttribute('y', svg.height.animVal.value.toFixed(2) - 12);
					timeMove.innerHTML = priceArray[index].fullTime;
					elId('open-price').innerText = priceArray[index].open;
					elId('highest-price').innerText  = priceArray[index].high;
					elId('lowest-price').innerText  = priceArray[index].low;
					elId('current-price').innerText  = priceArray[index].close;
					return;
				 }
				svg.innerHTML += `<text id="timeMove" x="${vt-40}" y="${svg.height.animVal.value.toFixed(2) - 12}" style="fill: white">${priceArray[index].fullTime}</text>`; 
				
			}
			chartMainDiv.addEventListener('click', pointerclick);
}
function pointerclick() {
		svg.removeEventListener('pointermove', pointermov);
		elId('crosshairy').remove();
		elId('crosshairv').remove();
		elId('timeMove').remove();
		elId('open-price').innerText = priceArray[len].open;
		elId('highest-price').innerText  = priceArray[len].high;
		elId('lowest-price').innerText  = priceArray[len].low;
		elId('current-price').innerText  = priceArray[len].close;
					
		chartMainDiv.removeEventListener('click', pointerclick);
		cursor.classList.add('active-toolBtn');
		crosshair.classList.remove('active-toolBtn');
		tooltipFunc();
}
crosshair.onclick = () => {
	svg.addEventListener('pointerenter', drawCrosshair);
}

/* Show context menu */

let closeBtn1 = elId('close-btn1');
let closeBtn2 = elId('close-btn2');
let context_menu_div = elId('context-menu');
function closeContextMenu() {
	context_menu_div.style.display = "none";
	modal.style.display = 'none';
}
closeBtn1.addEventListener('click', closeContextMenu);
closeBtn2.addEventListener('click', closeContextMenu);

/* Show and hide chart objects list / context menu */

let objects_list = elId('objects-list');
let modal = elId('cover-div');
try { objects =  JSON.parse(localStorage.getItem('objects')) || []; }catch(e){};
(function() {
	let storedObj = JSON.parse(localStorage.getItem('objects')) || [];
	for(let val of storedObj) {
		if(val.name == "Circle") {
			svg.innerHTML += `<circle id="${val.id}" cx="${val.cx}" cy="${val.cy}" r="${val.r}" style="fill: rgb(138, 32, 93, 0.737); stroke: none; stroke-width: 2" />`;
		}else if(val.name == 'Rectangle') {
			svg.innerHTML += `<rect id="${val.id}" x="${val.x}" y="${val.y}" width="${val.w}" height="${val.h}" style="fill: rgb(138, 32, 93, 0.737); stroke: none; stroke-width: 2" />`;
		}else if(val.name == 'Vertical line') {
			svg.innerHTML += `<g stroke="rgb(255, 255, 255)" class="line"><path id="${val.id}" stroke-width="1" d="M${val.m} 0 L${val.l} ${val.l2}" /></g>`; 
		}else if(val.name == 'Horizontal line') {
			svg.innerHTML += `<g stroke="rgb(255, 255, 255)" class="line"><path id="${val.id}" stroke-width="1" d="M0 ${val.m} L${val.l} ${val.m}" /></g>`;
		}else if(val.name == 'Trend line') {
			svg.innerHTML += `<g stroke="rgb(255, 255, 255)"><path id="trendline-${val.id}" stroke-width="1" d="M${val.o} ${val.k} L${val.l} ${val.l2}" class="line" /></g>`;
		}
	}
})()
localStorage.removeItem
function showContMenu() {
	modal.style.display = 'block';
	context_menu_div.style.display = 'flex';
	objects_list.innerHTML = '';
	for(let i= 0; i< objects.length; i++) {
		let li = document.createElement('li');
		li.innerText = objects[i].name;
		li.id = objects[i].id;
		li.dataset.status = 'in_active';
		objects_list.appendChild(li);
	}
}

objects_list.addEventListener('click', function(e) {
	let li = e.target;
	if(li.tagName.toLowerCase() == 'li') {
		for(let key of objects_list.children) {
			if(key.dataset.status == 'active') {
				key.dataset.status = 'in_active';
			}
		}
		li.dataset.status = 'active';
	}
});
let removeObj = elId('remove-object');
function removeChartObject() {
	let lis = objects_list.children;
	if(objects_list && objects_list.childElementCount > 0)
	for(let i=0; i<objects.length; i++) {
		if(lis[i].dataset.status == 'active') {
			elId(objects[i].id).remove();
			lis[i].remove();
			delete objects[i];
			objects = objects.filter((obj) => obj !== undefined);
			localStorage.setItem('objects', JSON.stringify(objects));
		}
	}
}
removeObj.addEventListener('click', removeChartObject);

/* Shake context menu to signify modal */
modal.onclick = () => {
	context_menu_div.classList.add('shake');
	setTimeout(() => {
		context_menu_div.classList.remove('shake');
	}, 200);
}



/* Draw trendline on chart */
let ids = 0;
let trendline = elId('trendline');
let o, k, trendid=0, currentID = undefined,l,l2;
function drawLine(e) {
	if(trendline.classList.contains('active-toolBtn')) {
		o =chartDiv.scrollLeft + e.clientX, k = e.clientY-40; currentID = `trendline-${ids}`;
		svg.innerHTML += `<g stroke="rgb(255, 255, 255)"><path id="trendline-${ids}" stroke-width="1" d="M${o} ${k} L${o} ${k}" class="line" /></g>`;
		this.addEventListener('pointermove', modifyLine);
	}
}
function modifyLine(e) {
	svg.removeEventListener('pointerdown', drawLine);
	let drawnLine = elId(currentID);
	l= chartDiv.scrollLeft + e.clientX; l2 = e.clientY-40;
	drawnLine.setAttribute('d', `M${o} ${k} L${l} ${l2}`);
	this.addEventListener('pointerup', drawl);
}
function drawl(e) {
	svg.removeEventListener('pointermove', modifyLine);
	svg.removeEventListener('pointerup', drawl);
	svg.style.cursor = 'default';
	cursor.classList.add('active-toolBtn');
	trendline.classList.remove('active-toolBtn');
	let storedObj = JSON.parse(localStorage.getItem('objects')) || [];
	storedObj.push({
		name: "Trend line", 
		id: `trendline-${ids++}`,
		o,k,l,l2,
	}); objects = storedObj;
	localStorage.setItem('objects', JSON.stringify(storedObj));
	o=''; k='';
}

trendline.onclick = () => {
	svg.addEventListener('pointerdown', drawLine);
	svg.style.cursor = "crosshair";
}

/* Draw horizontal line on chart */

let horiz = elId('horiz');
let horizontalLine = {
	hz: 0,
	l: undefined,
	drawHorizLine(e) {
		if(horiz.classList.contains('active-toolBtn')) {
			this.hz = e.clientY-42; this.l = svg.width.animVal.value;
			svg.innerHTML += `<g stroke="rgb(255, 255, 255)" class="line"><path id="horizontalline-${ids}" stroke-width="1" d="M0 ${this.hz} L${this.l} ${this.hz}" /></g>`;
			svg.removeEventListener('pointerdown', this.drawHorizLine);
			svg.style.cursor = 'default';
		cursor.classList.add('active-toolBtn');
		horiz.classList.remove('active-toolBtn');
		let storedObj = JSON.parse(localStorage.getItem('objects')) || [];
		storedObj.push({
			name: "Horizontal line", 
			id: `horizontalline-${ids++}`,
			m: this.hz, 
			l: this.l,
		}); objects = storedObj;
		localStorage.setItem('objects', JSON.stringify(storedObj));
		this.hz='';
		}
	}
}
horiz.onclick = () => {
	svg.addEventListener('pointerdown', horizontalLine.drawHorizLine);
	svg.style.cursor = "crosshair";
}

/* Draw vertical line on chart */

let vertic = elId('vertical');
let verticalLine = {
	vtc: 0,
	l2: undefined,
	drawVerticLine(e) {
		if(vertic.classList.contains('active-toolBtn')) {
			this.vtc = chartDiv.scrollLeft + e.clientX; this.l2 = svg.height.animVal.value -25;
			svg.innerHTML += `<g stroke="rgb(255, 255, 255)" class="line"><path id="verticalline-${ids}" stroke-width="1" d="M${this.vtc} 0 L${this.vtc} ${this.l2}" /></g>`; 
			svg.removeEventListener('pointerdown', this.drawVerticLine);
			svg.style.cursor = 'default';
			cursor.classList.add('active-toolBtn');
			vertic.classList.remove('active-toolBtn');
			let storedObj = JSON.parse(localStorage.getItem('objects')) || [];
			storedObj.push({
				name: "Vertical line", 
				id: `verticalline-${ids++}`,
				m: this.vtc, 
				l: this.vtc,
				l2: this.l2,
			}); objects = storedObj;
			localStorage.setItem('objects', JSON.stringify(storedObj));
			this.vtc = '';
		}
	}
};
vertic.onclick = () => {
	svg.addEventListener('pointerdown', verticalLine.drawVerticLine);
	svg.style.cursor = "crosshair";
}

let removeMinus = (num) => (String(num)[0] == '-')? Number(String(num).slice(1)) : num;
/* Draw rectangle on chart */ 

let rectangle = elId('rectangle');
let rec, rec2, rectID = undefined, w,h;

function drawRectangle(e) {
	if(rectangle.classList.contains('active-toolBtn')) {
		rec = chartDiv.scrollLeft + e.clientX; rec2 = e.clientY-40; rectID = `rectangle-${ids}`
		svg.innerHTML += `<rect id="${rectID}" x="${rec}" y="${rec2}" width="${(chartDiv.scrollLeft + e.clientX)-rec}" height="${(e.clientY-40) - rec2}" style="fill: rgb(138, 32, 93, 0.737); stroke: none; stroke-width: 2; " />`;
		this.addEventListener('pointermove', drawRect);
	}
}
function drawRect(e) {
	let getRect = elId(rectID);
	getRect.setAttribute('width', `${(chartDiv.scrollLeft + e.clientX)-rec}`);
	getRect.setAttribute('height', `${removeMinus((e.clientY-40) - rec2)}`);
	svg.addEventListener('pointerup', drawR);
	w = (chartDiv.scrollLeft + e.clientX)-rec;
	h = removeMinus((e.clientY-40) - rec2);
}
function drawR() {
	svg.removeEventListener('pointerdown', drawRectangle);
	svg.removeEventListener('pointermove', drawRect);
	svg.removeEventListener('pointerup', drawR);
	svg.style.cursor = 'default';
	cursor.classList.add('active-toolBtn');
	rectangle.classList.remove('active-toolBtn');
	let storedObj = JSON.parse(localStorage.getItem('objects')) || [];
	storedObj.push({
		name: "Rectangle", 
		id: `${rectID}`,
		x: rec, 
		y: rec2, w, h
	}); ids++; objects = storedObj;
	localStorage.setItem('objects', JSON.stringify(storedObj));
}
rectangle.onclick = () => {
	svg.addEventListener('pointerdown', drawRectangle);
	svg.style.cursor = "crosshair";
};

let circle = elId('circle');
let circ,circ2,circID,cy;
 
function drawCircle(e) {
	if(circle.classList.contains('active-toolBtn')) {
		circ = chartDiv.scrollLeft + e.clientX; circ2 = e.clientY-40; circID = `circle-${ids}`
		svg.innerHTML += `<circle id="${circID}" cx="${circ}" cy="${circ2}" r="${(chartDiv.scrollLeft + e.clientX)-circ}" style="fill: rgb(138, 32, 93, 0.737); stroke: none; stroke-width: 2; " />`;
		this.addEventListener('pointermove', drawCirc);
	}
}
function drawCirc(e) {
	let getCirc = elId(circID);
	getCirc.setAttribute('r', `${removeMinus((e.clientY-40) - circ2)}`);
	svg.addEventListener('pointerup', drawC);
	cy = (e.clientY-40) - circ2;
} 
function drawC() {
	svg.removeEventListener('pointerdown', drawCircle);
	svg.removeEventListener('pointermove', drawCirc);
	svg.removeEventListener('pointerup', drawC);
	svg.style.cursor = 'default';
	cursor.classList.add('active-toolBtn');
	circle.classList.remove('active-toolBtn');
	let storedObj = JSON.parse(localStorage.getItem('objects')) || [];
	storedObj.push({
		name: "Circle", 
		id: `${circID}`,
		cx: circ, 
		cy: circ2,
		r: cy
	}); ids; objects = storedObj;
	localStorage.setItem('objects', JSON.stringify(storedObj));
	localStorage.setItem('ids', ids++);
}

circle.onclick = () => {
	svg.addEventListener('pointerdown', drawCircle)
	svg.style.cursor = "crosshair";
};
