<?php include 'header.php' ?>
<body>
<!-- Left Panel -->

<?php include 'sidebar.php' ?>

<!-- Left Panel -->

<!-- Right Panel -->

<div id="right-panel" class="right-panel">

    <!-- Header-->
    <header id="header" class="header">

        <div class="header-menu">

            <div class="col-sm-4">
                <a id="menuToggle" class="menutoggle pull-left"><i class="fa fa fa-tasks"></i></a>
                <div class="header-left">
                    <div class="dropdown for-notification">
                        <button class="btn btn-secondary dropdown-toggle" type="button" id="notification"
                                data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                            <i class="fa fa-bell"></i>
                            <span class="count bg-danger">0</span>
                        </button><!--
                            <div class="dropdown-menu" aria-labelledby="notification">
                                <p class="red">You have 3 Notification</p>
                                <a class="dropdown-item media bg-flat-color-1" href="#">
                                <i class="fa fa-check"></i>
                                <p>Server #1 overloaded.</p>
                            </a>
                                <a class="dropdown-item media bg-flat-color-4" href="#">
                                <i class="fa fa-info"></i>
                                <p>Server #2 overloaded.</p>
                            </a>
                                <a class="dropdown-item media bg-flat-color-5" href="#">
                                <i class="fa fa-warning"></i>
                                <p>Server #3 overloaded.</p>
                            </a>
                            </div>-->
                    </div>
                </div>

            </div>


            <div class="col-sm-8">
                <p style="font-size:24px; color:#444444; font-variant:small-caps; font-weight:450; margin-bottom:0px; margin-top:12px; ">
                    Crop Disease Monitoring using AI Model </p>

            </div>
        </div>

    </header><!-- /header -->
    <!-- Header-->

    <div class="breadcrumbs">
        <div class="col-sm-4">
            <div class="page-header float-left">
                <div class="page-title">
                    <h1>Dashboard</h1>
                </div>
            </div>
        </div>
        <div class="col-sm-8">
            <div class="page-header float-right">
                <div class="page-title">
                    <ol class="breadcrumb text-right">
                        <li><a href="#">Dashboard</a></li>
                        <li class="active">RUST</li>
                    </ol>
                </div>
            </div>
        </div>
    </div>

    <!-- <div class="col-lg-6">
		<div class="row form-group">
			<div class="col col-md-3"><label for="select" class=" form-control-label">Select Date:</label></div>
			<div class="col-12 col-md-6">
                <input id="selectedDate" name="selectedDate" type="date">
			</div>
            <div class="col-12 col-md-3">
                <button id="GoButton" style="margin:auto; color:#111111"> Go </button>
			</div>
		</div>
	</div> -->
                    
    <!-- <div class="row" style="padding-bottom:12px;">
	    <button id="GoButton" style="margin:auto; color:#111111"> Go </button>
    </div>                     -->

    <div class="content mt-3">
        <div class="animated fadeIn">


            <div class="row">
                <div class="col-lg-6">
                    <div class="card">
                        <div class="card-body">
                            <h4 class="mb-3">Choose an image from computer</h4>
                            <div class="flot-container">
                                
                            <form action="upload.php" method="post" enctype="multipart/form-data">
                            Select image to upload:
                            <input type="file" name="fileToUpload" id="fileToUpload">
                            <!-- <input type="submit" value="Upload Image" name="submit"> -->
                            <br>
                            <button style="text-align:center;" class="btn btn-primary" id="capture-btn">Upload Image</button>
    
                        </form>

                            </div>
                        </div>
                    </div><!-- /# card -->
                </div><!-- /# column -->

                <!-- <div class="col-lg-2">
                    <p>OR</p>
                </div> -->
                <div class="col-lg-6">
                    <div class="card">
                        <div class="card-body">
                            <h4 class="mb-3">Capture image using camera</h4>
                            <!-- <div class="flot-container"> -->
                                
                            <!-- <div id="newImages"></div> -->
<!-- <div class="row"> -->
    <!-- <div class="cell"> -->
      <video id="player" width="100%" height="250px" autoplay></video>
    <!-- </div> -->
    <!-- <div class="cell"></div>
      <canvas id="canvas" ></canvas>
    </div> -->
  <!-- </div> -->

  <div id="pick-image">
    <label>Camera is not supported. Upload an Image.</label>
    
  </div>

  <!-- <div class="center"> -->
    <button class="btn btn-primary" id="capture-btn">Capture</button>
  <!-- </div> -->
  

                            <!-- </div> -->
                        </div>
                    </div><!-- /# card -->
                </div><!-- /# column -->
            </div><!-- /# row -->


            <div class="row">
                <div class="col-lg-12">
                    <div class="card">
                        <div class="card-body">
                            <h4> Result </h4>
                        </div>
                    </div>
                </div>
            </div>

            

            <script type="text/javascript">
                $(document).ready(function() {
                    for (let counter = 2; counter <= 9; counter++) {
                        $(`#node${counter}`).css('display', 'none');
                    }
                })

                $('#node').on('click', function () {
                    var node_no = $(this).val();
                    for (let counter = 1; counter <= 9; counter++) {
                        if (counter !== node_no) {
                            $(`#node${counter}`).css('display', 'none');
                        }
                    }
                    $(`#node${node_no}`).css('display', 'block');
                });
            </script>

            

                
            </div><!-- /# row -->

        </div><!-- .animated -->
    </div><!-- .content -->


</div><!-- /#right-panel -->

<!-- Right Panel -->


<script src="vendors/jquery/dist/jquery.min.js"></script>
<script src="vendors/popper.js/dist/umd/popper.min.js"></script>
<script src="vendors/bootstrap/dist/js/bootstrap.min.js"></script>
<script src="assets/js/main.js"></script>

<!--  flot-chart js -->
<script src="vendors/flot/excanvas.min.js"></script>
<script src="vendors/flot/jquery.flot.js"></script>
<script src="vendors/flot/jquery.flot.pie.js"></script>
<script src="vendors/flot/jquery.flot.time.js"></script>
<script src="vendors/flot/jquery.flot.stack.js"></script>
<script src="vendors/flot/jquery.flot.resize.js"></script>
<script src="vendors/flot/jquery.flot.crosshair.js"></script>
<script src="assets/js/init-scripts/flot-chart/curvedLines.js"></script>
<script src="assets/js/init-scripts/flot-chart/flot-tooltip/jquery.flot.tooltip.min.js"></script>
<script src="assets/js/init-scripts/flot-chart/flot-chart-AirMois.js"></script>
<script src="assets/js/init-scripts/flot-chart/jquery.flot.axislabels.js"></script>

<!-- <script language="JavaScript">
    Webcam.set({
        width: 490,
        height: 390,
        image_format: 'jpeg',
        jpeg_quality: 90
    });
  
    Webcam.attach( '#my_camera' );
  
    function take_snapshot() {
        Webcam.snap( function(data_uri) {
            $(".image-tag").val(data_uri);
            document.getElementById('results').innerHTML = '<img src="'+data_uri+'"/>';
        } );
    }
</script> -->



<script src="script.js"></script>

</body>

</html>
