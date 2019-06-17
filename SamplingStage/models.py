from django.db import models



class Ocean(models.Model):
    """
    This table is pure source data set
    """
    ASIN = models.CharField(max_length=30,null=False)
    on_sale = models.BooleanField(default=True)
    marketplace = models.IntegerField()
    risk_level = models.CharField(max_length=10)
    vendor_type = models.CharField(verbose_name='ACD|vcd', max_length=15)
    vendor_name = models.CharField(verbose_name='Vendor Code',max_length=15)
    factory_name = models.CharField(verbose_name='Factory Name', max_length=30)
    product_type = models.CharField(verbose_name='Product Type', max_length=30)
    parent_asin = models.CharField(verbose_name= 'Parent Asin', max_length=30)
    brand_name = models.CharField(verbose_name='Brand Name', max_length=30)
    description = models.TextField(verbose_name='Product Description')


class MFP(models.Model):
    """
    collecting infromation from agile based on ASINs from Ocean
    """
    ASIN = models.ForeignKey(Ocean,on_delete=models.CASCADE)
    mfp = models.TextField(verbose_name='Manufacturer Part Number',unique=True)
    pl_doc = models.TextField(verbose_name='PL Docs')
    region = models.CharField(verbose_name='Active Region', max_length= 30)
    concession = models.TextField(verbose_name='Cocession Scenario')
    approved_by = models.CharField(verbose_name='Approved By', max_length=30)
    additional_notes = models.TextField(verbose_name='Additional Notes')
    status = models.ForeignKey('ReviewStage.StatusAndNextStep',
                               on_delete= models.CASCADE,
                               related_name='mfp_status',
                               default='',
                               editable=False)  # mfp level status and next step
    next_step = models.ForeignKey('ReviewStage.StatusAndNextStep',on_delete= models.CASCADE,
                               related_name='mfp_next_step',
                               default='',
                               editable=False)







