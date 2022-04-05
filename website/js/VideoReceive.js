var ipServeur = "10.30.50.186"

function launchcam(){

    //var ws1 = new WebSocket("ws://10.30.50.185:5678/"); // Tu dois mettre l'addresse de la machine qui fait tourner le python
    var ws1 = new WebSocket("ws://localhost:8585/");
    ws1.onopen = function(){
        console.log("Message envoyé")
        ws1.send("Test message");

    }

    ws1.onmessage= function (event){
        console.log("Message recu : ")
        console.log(event.data);
    }
}
openSocket = () => {
    socket = new WebSocket("ws://"+ipServeur+":5678");
    let msg = document.getElementById("msg");
    socket.addEventListener('open', (e) => {
        document.getElementById("status").innerHTML = "Opened";
    });
    socket.addEventListener('message', (e) => {
        let info = e.data
        console.log(info)
        if(typeof(info) === "string"){
            document.getElementById("spanVitesse").textContent=info;
        }else{
            let ctx = msg.getContext("2d");
            let image = new Image();
            image.src = URL.createObjectURL(e.data);
            image.addEventListener("load", (e) => {
                ctx.drawImage(image, 0, 0, msg.width, msg.height);
            });
        }
    });
}

function switchmode(){
    var ws1 = new WebSocket("ws://"+ipServeur+":5679");
    //var ws1 = new WebSocket("ws://localhost:8585/");
    content = document.getElementById("spanMode").textContent
    if(content === "Mode actuel : Autonome"){
        ws1.onopen = function(){
            console.log("Message envoyé : manual")
            ws1.send("manual");
        }
        document.getElementById("spanMode").textContent="Mode actuel : Manuel"
    }else{
        ws1.onopen = function(){
            console.log("Message envoyé : autonomous")
            ws1.send("autonomous");
        }
        document.getElementById("spanMode").textContent="Mode actuel : Autonome"
    }
}

function modePark(){
    var ws1 = new WebSocket("ws://"+ipServeur+":5679");
    //var ws1 = new WebSocket("ws://localhost:8585/");
    content = document.getElementById("parkMode").textContent
    if(content === "Park mode : Off"){
        ws1.onopen = function(){
            console.log("Message envoyé : on")
            ws1.send("Park mode : on")
        }
        document.getElementById("parkMode").textContent="Park mode : On"
    }else{
        ws1.onopen = function(){
            console.log("Message envoyé : off")
            ws1.send("Park mode : off")
        }
        document.getElementById("parkMode").textContent="Park mode : Off"
    }
}
