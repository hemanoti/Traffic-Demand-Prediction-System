// Traffic Demand

new Chart(document.getElementById("trafficChart"),{

type:"line",

data:{

labels:["6 AM","9 AM","12 PM","3 PM","6 PM","9 PM"],

datasets:[{

label:"Traffic Volume",

data:[120,480,350,620,820,450],

borderColor:"#22c55e",

backgroundColor:"rgba(34,197,94,.2)",

fill:true,

tension:.4

}]

},

options:{

responsive:true

}

});


// Vehicle Distribution

new Chart(document.getElementById("vehicleChart"),{

type:"doughnut",

data:{

labels:["Cars","Bikes","Buses","Trucks"],

datasets:[{

data:[45,25,15,15],

backgroundColor:[

"#22c55e",

"#2563eb",

"#f59e0b",

"#8b5cf6"

]

}]

}

});


// Congestion

new Chart(document.getElementById("congestionChart"),{

type:"bar",

data:{

labels:["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],

datasets:[{

label:"Congestion",

data:[60,70,80,70,92,60,42],

backgroundColor:"#8b5cf6"

}]

}

});