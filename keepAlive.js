function KeepAlive() {
   console.log("KeepAlive tick");
   document.querySelector("#connect-button")?.click();
}
setInterval(KeepAlive, 60000); // Clicks every 60 seconds
