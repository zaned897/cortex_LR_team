echo "========================================================================="
echo "INSALLING OS DEV TOOLS..."
echo "========================================================================="

# system utils
yum install gcc-c++ -y  # c++ compiler for tesseract make
yum install diffutils -y  # leptonica compiler tools
yum install automake -y 	# autoconfig tesseract util
yum install poppler-utils -y # pdf utils 
yum install -y file
yum install -y find


echo "========================================================================="
echo "INSALLING LEPTONICA FOR OCR..."
echo "========================================================================="

# leptonica image proc for tesseract 
yum install -y libtool libcurl libjpeg-devel libpng-devel libtiff-devel zlib-devel && \
cd /home/app/src/leptonica-1.80.0/ && ./configure && make && make install # install leptonica for tesseract
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig 	# add lept.ce to path, if not tesseract cant make since version mistake
export LD_LIBRARY_PATH=/usr/local/lib

echo "========================================================================="
echo "INSALLING TESSERACT OCR.."
echo "========================================================================="

# tesseract ocr (opticall character recognition)

yum install cairo pango icu4c autoconf libffi libarchive libpng && \
cd /home/app/src/tesseract-4.1.1/ && ./autogen.sh && ./configure && make && make install && ldconfig # install tesseract (OPTICAL CHARACTERS RECOGNITION)
cp /home/app/src/eng.traineddata /usr/local/share/tessdata/ # copy language to work (OCR)


echo "========================================================================="
echo "INSTALLING DATABASE..."
echo "========================================================================="

yum install -y https://download.postgresql.org/pub/repos/yum/12/redhat/rhel-6-x86_64/postgresql12-libs-12.3-1PGDG.rhel6.x86_64.rpm
yum install -y https://download.postgresql.org/pub/repos/yum/12/redhat/rhel-6-x86_64/postgresql12-12.3-1PGDG.rhel6.x86_64.rpm
yum install -y https://download.postgresql.org/pub/repos/yum/12/redhat/rhel-6-x86_64/postgresql12-server-12.3-1PGDG.rhel6.x86_64.rpm
yum install -y https://download.postgresql.org/pub/repos/yum/12/redhat/rhel-6-x86_64/postgresql12-devel-12.3-1PGDG.rhel6.x86_64.rpm
yum install -y https://download.postgresql.org/pub/repos/yum/12/redhat/rhel-6-x86_64/postgresql12-contrib-12.3-1PGDG.rhel6.x86_64.rpm
yum install -y https://download.postgresql.org/pub/repos/yum/12/redhat/rhel-6-x86_64/postgresql12-docs-12.3-1PGDG.rhel6.x86_64.rpm


echo "========================================================================="
echo "INSALLING PYTHON DEPENDENCIES..."
echo "========================================================================="
