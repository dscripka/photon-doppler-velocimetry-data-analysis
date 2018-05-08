// Javascript code to allow downloading of velocity trace(s)
if (typeof s1 === 'undefined') {
    var data = source.data;
    var filetext = 'analysis parameters (fft window/wavelet width, fft/wavelet frequency step, timestep): ' + '\n' + 'time (s), velocity (m/s)\n';  //define the header for the csv file
    for (i=0; i < data['velocity'].length; i++) {
        var currRow = [data['x'][i].toString(),
                       data['velocity'][i].toString() + '\n'];
    
        var joined = currRow.join(', ');
        filetext = filetext.concat(joined);
    }
}

// else if (typeof source === 'undefined') {
//     var fft_data = s1.data
//     var wvt_data = s2.data
//     var zero_crossing_data = s3.data
//     var filetext = 'STFT Parameters: \nWVT Paramters: \n Zero Crossing Parameters: ' + '\n' + 'time (s), stft velocity (m/s), wvt velocity (m/s), zero crossing velocity (m/s)\n';  //define the header for the csv file
//     for (i=0; i < fft_data['x'].length; i++) {
//         var currRow = [fft_data['x'][i].toString(),
//                        fft_data['velocity'][i].toString(),
//                        wvt_data['velocity'][i].toString(),
//                        //zero_crossing_data['velocity'][i].toString() + '\n' //This has different spacing than the other data sets
//                       ];
    
//         var joined = currRow.join(', ');
//         filetext = filetext.concat(joined);
//     }
// }

var filename = 'velocity.csv';
var blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' });

//addresses IE users
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob, filename);
}
// other browsers
else {
    var link = document.createElement("a");
    link = document.createElement('a')
    link.href = URL.createObjectURL(blob);
    link.download = filename
    link.target = "_blank";
    link.style.visibility = 'hidden';
    link.dispatchEvent(new MouseEvent('click'))
}