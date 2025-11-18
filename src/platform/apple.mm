#include "apple.hh"

#import <Foundation/Foundation.h>
#include <dispatch/dispatch.h>

#ifdef JST_OS_IOS
#import <UIKit/UIKit.h>
#else
#import <AppKit/AppKit.h>
#endif
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

#ifdef JST_OS_IOS
@interface JSTDocumentPickerDelegate : NSObject <UIDocumentPickerDelegate>
@property(nonatomic, assign) dispatch_semaphore_t semaphore;
@property(nonatomic, assign) std::string* outPath;
@property(nonatomic, assign) Result* outResult;
@end

static NSMutableSet<JSTDocumentPickerDelegate*>* JSTActivePickerDelegates() {
    static NSMutableSet<JSTDocumentPickerDelegate*>* delegates = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        delegates = [NSMutableSet set];
    });
    return delegates;
}

static void JSTRegisterDocumentPickerDelegate(JSTDocumentPickerDelegate* delegate) {
    [JSTActivePickerDelegates() addObject:delegate];
}

static void JSTUnregisterDocumentPickerDelegate(JSTDocumentPickerDelegate* delegate) {
    [JSTActivePickerDelegates() removeObject:delegate];
}

static UIViewController* JSTTopViewController() {
    __block UIViewController* rootController = nil;
    if (@available(iOS 13.0, *)) {
        for (UIScene* scene in [UIApplication sharedApplication].connectedScenes) {
            if (scene.activationState != UISceneActivationStateForegroundActive ||
                ![scene isKindOfClass:[UIWindowScene class]]) {
                continue;
            }
            UIWindowScene* windowScene = (UIWindowScene*)scene;
            for (UIWindow* window in windowScene.windows) {
                if (window.isKeyWindow) {
                    rootController = window.rootViewController;
                    break;
                }
            }
            if (rootController != nil) {
                break;
            }
        }
    }

    if (rootController == nil) {
        rootController = [UIApplication sharedApplication].keyWindow.rootViewController;
        if (rootController == nil) {
            rootController = [UIApplication sharedApplication].windows.firstObject.rootViewController;
        }
    }

    UIViewController* topController = rootController;
    while (topController.presentedViewController != nil) {
        topController = topController.presentedViewController;
    }

    return topController;
}

API_AVAILABLE(ios(14.0)) static NSArray<UTType*>* JSTContentTypesForExtensions(
    const std::vector<std::string>& extensions) {
    NSMutableArray<UTType*>* contentTypes = [NSMutableArray array];
    for (const auto& ext : extensions) {
        NSString* nsExt = [NSString stringWithUTF8String:ext.c_str()];
        UTType* type = [UTType typeWithFilenameExtension:nsExt];
        if (type != nil) {
            [contentTypes addObject:type];
        }
    }

    if ([contentTypes count] == 0) {
        return @[[UTType item]];
    }

    return contentTypes;
}

@implementation JSTDocumentPickerDelegate

- (void)completeWithURL:(NSURL*)url success:(BOOL)success {
    if (success && url != nil && self.outPath != nullptr) {
        BOOL needsAccess = [url startAccessingSecurityScopedResource];
        NSString* filePath = [url path];
        if (filePath != nil) {
            *(self.outPath) = std::string([filePath UTF8String]);
        }
        if (needsAccess) {
            [url stopAccessingSecurityScopedResource];
        }
    } else if (!success) {
        JST_ERROR("Cannot pick file.");
    }

    if (self.outResult != nullptr) {
        *(self.outResult) = success ? Result::SUCCESS : Result::ERROR;
    }

    if (self.semaphore != nullptr) {
        dispatch_semaphore_signal(self.semaphore);
    }

    JSTUnregisterDocumentPickerDelegate(self);
}

- (void)documentPicker:(UIDocumentPickerViewController*)controller didPickDocumentsAtURLs:(NSArray<NSURL*>*)urls {
    NSURL* url = urls.firstObject;
    [self completeWithURL:url success:(url != nil)];
    [controller dismissViewControllerAnimated:YES completion:nil];
}

- (void)documentPickerWasCancelled:(UIDocumentPickerViewController*)controller {
    if (self.outResult != nullptr) {
        *(self.outResult) = Result::ERROR;
    }
    if (self.semaphore != nullptr) {
        dispatch_semaphore_signal(self.semaphore);
    }
    JST_ERROR("User cancelled file picker.");
    [controller dismissViewControllerAnimated:YES completion:nil];
    JSTUnregisterDocumentPickerDelegate(self);
}

@end
#endif  // JST_OS_IOS

namespace Jetstream::Platform {

Result OpenUrl(const std::string& url) {
    NSString* nsUrl = [NSString stringWithUTF8String:url.c_str()];
    NSURL* urlObj = [NSURL URLWithString:nsUrl];

    if (!urlObj) {
        JST_ERROR("Cannot open URL because it's invalid.");
        return Result::ERROR;
    }

#ifdef JST_OS_IOS
    if ([[UIApplication sharedApplication] canOpenURL:urlObj]) {
        [[UIApplication sharedApplication] openURL:urlObj options:@{} completionHandler:nil];
    } else {
        JST_ERROR("Cannot open URL.");
        return Result::ERROR;
    }
#else
    if (![[NSWorkspace sharedWorkspace] openURL:urlObj]) {
        JST_ERROR("Cannot open URL.");
        return Result::ERROR;
    }
#endif

    return Result::SUCCESS;
}

Result PickFile(std::string& path, const std::vector<std::string>& extensions) {
    __block Result result = Result::ERROR;

    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);

    dispatch_async(dispatch_get_main_queue(), ^{
#ifdef JST_OS_MAC
        NSOpenPanel* panel = [NSOpenPanel openPanel];
        [panel setCanChooseFiles:YES];
        [panel setCanChooseDirectories:NO];
        [panel setAllowsMultipleSelection:NO];

        if (!extensions.empty()) {
            NSMutableArray* allowedTypes = [NSMutableArray array];
            for (const auto& ext : extensions) {
                NSString* nsExt = [NSString stringWithUTF8String:ext.c_str()];
                UTType* type = [UTType typeWithFilenameExtension:nsExt];
                if (type) {
                    [allowedTypes addObject:type];
                }
            }
            if ([allowedTypes count] > 0) {
                [panel setAllowedContentTypes:allowedTypes];
            }
        }
        // If extensions is empty, don't set allowedContentTypes to allow all file types

        if ([panel runModal] == NSModalResponseOK) {
            NSURL* url = [[panel URLs] objectAtIndex:0];
            NSString* filePath = [url path];
            path = std::string([filePath UTF8String]);
            result = Result::SUCCESS;
        } else {
            JST_ERROR("Cannot pick file.");
            result = Result::ERROR;
        }
#elif defined(JST_OS_IOS)
        if (!@available(iOS 14.0, *)) {
            JST_ERROR("File picker requires iOS 14 or later.");
            result = Result::ERROR;
            dispatch_semaphore_signal(semaphore);
            return;
        }

        UIViewController* presenter = JSTTopViewController();
        if (presenter == nil) {
            JST_ERROR("Cannot present file picker.");
            result = Result::ERROR;
            dispatch_semaphore_signal(semaphore);
            return;
        }

        NSArray<UTType*>* types = JSTContentTypesForExtensions(extensions);
        UIDocumentPickerViewController* picker =
            [[UIDocumentPickerViewController alloc] initForOpeningContentTypes:types];
        JSTDocumentPickerDelegate* delegate = [[JSTDocumentPickerDelegate alloc] init];
        delegate.semaphore = semaphore;
        delegate.outPath = &path;
        delegate.outResult = &result;
        picker.delegate = delegate;
        picker.shouldShowFileExtensions = YES;
        picker.allowsMultipleSelection = NO;
        JSTRegisterDocumentPickerDelegate(delegate);
        [presenter presentViewController:picker animated:YES completion:nil];
        return;
#endif
        dispatch_semaphore_signal(semaphore);
    });

    dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);

    return result;
}

Result PickFolder(std::string& path) {
    __block Result result = Result::ERROR;

    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);

    dispatch_async(dispatch_get_main_queue(), ^{
#ifdef JST_OS_MAC
        NSOpenPanel* panel = [NSOpenPanel openPanel];
        [panel setCanChooseFiles:NO];
        [panel setCanChooseDirectories:YES];
        [panel setAllowsMultipleSelection:NO];

        if ([panel runModal] == NSModalResponseOK) {
            NSURL* url = [[panel URLs] objectAtIndex:0];
            NSString* folderPath = [url path];
            path = std::string([folderPath UTF8String]);
            result = Result::SUCCESS;
        } else {
            JST_ERROR("Cannot pick folder.");
            result = Result::ERROR;
        }
#elif defined(JST_OS_IOS)
        if (!@available(iOS 14.0, *)) {
            JST_ERROR("Folder picker requires iOS 14 or later.");
            result = Result::ERROR;
            dispatch_semaphore_signal(semaphore);
            return;
        }

        UIViewController* presenter = JSTTopViewController();
        if (presenter == nil) {
            JST_ERROR("Cannot present folder picker.");
            result = Result::ERROR;
            dispatch_semaphore_signal(semaphore);
            return;
        }

        NSArray<UTType*>* folders = @[ [UTType folder] ];
        UIDocumentPickerViewController* picker =
            [[UIDocumentPickerViewController alloc] initForOpeningContentTypes:folders];
        JSTDocumentPickerDelegate* delegate = [[JSTDocumentPickerDelegate alloc] init];
        delegate.semaphore = semaphore;
        delegate.outPath = &path;
        delegate.outResult = &result;
        picker.delegate = delegate;
        picker.shouldShowFileExtensions = YES;
        picker.allowsMultipleSelection = NO;
        JSTRegisterDocumentPickerDelegate(delegate);
        [presenter presentViewController:picker animated:YES completion:nil];
        return;
#endif
        dispatch_semaphore_signal(semaphore);
    });

    dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);

    return result;
}

Result SaveFile(std::string& path) {
    __block Result result = Result::ERROR;

    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);

    dispatch_async(dispatch_get_main_queue(), ^{
#ifdef JST_OS_MAC
        NSSavePanel* panel = [NSSavePanel savePanel];

        if ([panel runModal] == NSModalResponseOK) {
            NSURL* url = [panel URL];
            NSString* filePath = [url path];
            path = std::string([filePath UTF8String]);
            result = Result::SUCCESS;
        } else {
            JST_ERROR("Cannot save file.");
            result = Result::ERROR;
        }
#elif defined(JST_OS_IOS)
        if (!@available(iOS 14.0, *)) {
            JST_ERROR("Save file picker requires iOS 14 or later.");
            result = Result::ERROR;
            dispatch_semaphore_signal(semaphore);
            return;
        }

        UIViewController* presenter = JSTTopViewController();
        if (presenter == nil) {
            JST_ERROR("Cannot present save file picker.");
            result = Result::ERROR;
            dispatch_semaphore_signal(semaphore);
            return;
        }

        const std::vector<std::string> saveExtensions = {"yaml", "yml"};
        NSArray<UTType*>* types = JSTContentTypesForExtensions(saveExtensions);
        UIDocumentPickerViewController* picker =
            [[UIDocumentPickerViewController alloc] initForOpeningContentTypes:types];
        picker.allowsDocumentCreation = YES;
        picker.shouldShowFileExtensions = YES;
        picker.allowsMultipleSelection = NO;

        JSTDocumentPickerDelegate* delegate = [[JSTDocumentPickerDelegate alloc] init];
        delegate.semaphore = semaphore;
        delegate.outPath = &path;
        delegate.outResult = &result;
        picker.delegate = delegate;
        JSTRegisterDocumentPickerDelegate(delegate);
        [presenter presentViewController:picker animated:YES completion:nil];
        return;
#endif
        dispatch_semaphore_signal(semaphore);
    });

    dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);

    return result;
}

}  // namespace Jetstream::Platform
