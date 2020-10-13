# cd ..

find .. -type d -name out_data -exec rm -rf {} \;
find .. -type d -name out_plot -exec rm -rf {} \;
find .. -type d -name out_plots -exec rm -rf {} \;
find .. -type d -name voro_temp -exec rm -rf {} \;
find .. -type d -name __old__ -exec rm -rf {} \;
find .. -type d -name __temp__ -exec rm -rf {} \;
find .. -type d -name __test__ -exec rm -rf {} \;
find .. -type d -name __bkp__ -exec rm -rf {} \;
