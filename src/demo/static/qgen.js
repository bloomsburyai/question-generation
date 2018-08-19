function ping()
{
  query = "/api/ping"
  $.ajax({
    url: query,
    cache: false
  })
    .done(function( html ) {
      if(html=='ack')
      {
        $( "#connection-indicator" ).attr('class', 'good');
      }
      else {
        $( "#connection-indicator" ).attr('class', 'bad');
      }

    })
    .fail(function(){
      $( "#connection-indicator" ).attr('class', 'bad');
    });
  setTimeout(ping, 10000);
}
function getQ()
{
  query = "/api/generate?context=" +encodeURIComponent($('#context').val())+ "&answer="+$('#answer').val()
  $( "#response-spinner" ).toggleClass('d-none');
  $( "#response" ).toggleClass('d-none');
  $.ajax({
    url: query,
    cache: false
  })
    .done(function( html ) {
      $( "#response" ).html("<p>"+html+"</p>");

      $( "#response-spinner" ).toggleClass('d-none');
      $( "#response" ).toggleClass('d-none');
    })
    .fail(function(){
      $( "#response" ).html("<p>There was an error generating a question :(</p>");

      $( "#response-spinner" ).toggleClass('d-none');
      $( "#response" ).toggleClass('d-none');
    });
}
function seed(c,a)
{
  $('#context').val(c);
  $('#answer').val(a);
}
function seedWithFootball()
{
  ctxt = "Harry Kane's stoppage-time winner ensured England started their World Cup campaign with victory after Tunisia threatened to snatch a point in Volgograd. Kane scored his second goal of the game with a clever header as Gareth Southgate's side recorded England's first win in the opening game of a major tournament since they beat Paraguay in the 2006 World Cup. England's captain gave them the reward they deserved for a brilliant start by turning in the opener in the 11th minute after Tunisia keeper Mouez Hassen, who went off injured in the first half, clawed out John Stones' header. England ran Tunisia ragged in that spell but were punished for missing a host of chances when Ferjani Sassi equalised from the penalty spot against the run of play after Kyle Walker was penalised for an elbow on Fakhreddine Ben Youssef. Tunisia dug in to frustrate England in the second half but Kane was the match-winner with a late header from Harry Maguire's flick, justice being done after referee Wilmar Roldan and the video assistant referee (VAR) had failed to spot him being wrestled to the ground twice in the penalty area. England play Panama, who lost 3-0 to Belgium earlier on Monday, in their next Group G game on Sunday, which kicks off at 13:00 BST and will be shown live on the BBC."
  ans = "Kyle Walker"
  seed(ctxt,ans)
}
function seedWithSquad()
{
  seed('Various princes of the Holy Land arrived in Limassol at the same time, in particular Guy de Lusignan. All declared their support for Richard provided that he support Guy against his rival Conrad of Montferrat. The local barons abandoned Isaac, who considered making peace with Richard, joining him on the crusade, and offering his daughter in marriage to the person named by Richard. But Isaac changed his mind and tried to escape. Richard then proceeded to conquer the whole island, his troops being led by Guy de Lusignan. Isaac surrendered and was confined with silver chains, because Richard had promised that he would not place him in irons. By 1 June, Richard had conquered the whole island. His exploit was well publicized and contributed to his reputation; he also derived significant financial gains from the conquest of the island. Richard left for Acre on 5 June, with his allies. Before his departure, he named two of his Norman generals, Richard de Camville and Robert de Thornham, as governors of Cyprus.', 'Guy de Lusignan')
}
function seedWithSquad2()
{
  seed('This in turn led to the establishment of the right-wing dictatorship of the Estado Novo under António de Oliveira Salazar in 1933. Portugal was one of only five European countries to remain neutral in World War II. From the 1940s to the 1960s, Portugal was a founding member of NATO, OECD and the European Free Trade Association (EFTA). Gradually, new economic development projects and relocation of mainland Portuguese citizens into the overseas provinces in Africa were initiated, with Angola and Mozambique, as the largest and richest overseas territories, being the main targets of those initiatives. These actions were used to affirm Portugal\'s status as a transcontinental nation and not as a colonial empire', 'António de Oliveira Salazar');
}
function seedWithSquad3()
{
  seed('with 4:51 left in regulation , carolina got the ball on their own 24-yard line with a chance to mount a game-winning drive , and soon faced 3rd-and-9 . on the next play , miller stripped the ball away from newton , and after several players dove for it , it took a long bounce backwards and was recovered by ward , who returned it five yards to the panthers 4-yard line . although several players dove into the pile to attempt to recover it , newton did not and his lack of aggression later earned him heavy criticism . meanwhile , denver \'s offense was kept out of the end zone for three plays , but a holding penalty on cornerback josh norman gave the broncos a new set of downs . then anderson scored on a 2-yard touchdown run and manning completed a pass to bennie fowler for a 2-point conversion , giving denver a 24–10 lead with 3:08 left and essentially putting the game away . carolina had two more drives , but failed to get a first down on each one .','miller')
}
function getCurrentModelSlug()
{
  query = "/api/model_current"
  $.ajax({
    url: query,
    cache: false
  })
    .done(function( html ) {
      $( "#model-slug" ).html("<i>"+html+"</i>");
    })
    .fail(function(){
      $( "#model-slug" ).html("**There was an error getting current model**");
    });

}
